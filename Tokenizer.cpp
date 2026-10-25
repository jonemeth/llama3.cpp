#include <algorithm>
#include <fstream>
#include <numeric>

#include "Tokenizer.h"

void TokenQueue::push(int token) { std::list<int>::push_back(token); }
int TokenQueue::pop()
{
    int token = std::list<int>::front();
    std::list<int>::pop_front();
    return token;
}

bool Tokenizer::TokenIndex::operator<(Tokenizer::TokenIndex const &second) const { return *str < *second.str; }
bool Tokenizer::TokenIndex::operator==(Tokenizer::TokenIndex const &second) const { return *str == *second.str; }

Tokenizer::Tokenizer(std::string path, int vocabSize)
    : vocabSize(vocabSize),
      vocabScores(vocabSize)
{
    for (int i = 0; i < 256; i++)
    {
        bytePieces[i * 2] = (unsigned char)i;
        bytePieces[i * 2 + 1] = '\0';
    }

    std::ifstream inputStream(path, std::ios::binary);

    inputStream.read((char *)(&maxTokenLength), sizeof(maxTokenLength));

    for (int i = 0; i < vocabSize; i++)
    {
        inputStream.read((char *)(&vocabScores[i]), sizeof(vocabScores[i]));

        int len;
        inputStream.read((char *)(&len), sizeof(len));

        std::string str(len, ' ');
        inputStream.read((char *)(&str[0]), len * sizeof(str[0]));

        vocab.push_back(str);
    }

    for (int i = 0; i < vocabSize; i++)
        sortedVocab.insert({&(vocab[i]), i});
}

std::optional<std::string> Tokenizer::decode(int token) const
{
    std::string piece = vocab[token];

    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1)
        piece = (char *)bytePieces + byte_val * 2;

    if (piece[0] == '\0')
        return std::nullopt;

    if (piece[1] == '\0' && !(isprint(piece[0]) || isspace(piece[0])))
        return std::nullopt;

    return piece;
}

TokenQueue Tokenizer::encode(std::string text, bool bos, bool eos) const
{
    TokenQueue tokens;

    if (bos)
        tokens.push(128000);

    std::string strBuffer;

    for (auto it = text.begin(); it != text.end(); ++it)
    {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the
        // rest 0x80 is 10000000 in UTF-8, all continuation bytes start with
        // "10" in first two bits so in English this is: "if this byte is not a
        // continuation byte"
        if ((*it & 0xC0) != 0x80)
            strBuffer.clear();

        // append the current byte to the buffer
        strBuffer += *it;

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning
        // str_buffer size.
        if (it < std::prev(text.end()) && (*std::next(it) & 0xC0) == 0x80 && strBuffer.length() < 4)
            continue;

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        std::optional<int> id = strLookUp(strBuffer);

        if (id)
            tokens.push(*id);
        else
            // byte_fallback encoding: just encode each byte as a token
            std::transform(strBuffer.begin(), strBuffer.end(), std::back_inserter(tokens),
                           [this](char c) { return strLookUp(std::string(1, c)).value(); });

        strBuffer.clear(); // protect against a sequence of stray UTF8 continuation bytes
    }

    merge(tokens);

    // add optional EOS (=128001) token, if desired
    if (eos)
        tokens.push(128001);

    return tokens;
}

std::optional<int> Tokenizer::strLookUp(std::string str) const
{
    auto it = sortedVocab.find(TokenIndex{&str, -1});
    if (it == sortedVocab.end())
        return std::nullopt;
    return it->id;
}

void Tokenizer::merge(TokenQueue &tokens) const
{
    struct MergeInfo
    {
        float score;
        TokenQueue::iterator start;
        int length;
        int newToken;
    };

    while (true)
    {
        std::optional<MergeInfo> mergeInfo;

        // merge the best consecutive pair or triple each iteration, according to the scores in vocab_scores
        for (int l = 2; l <= 3 && !mergeInfo.has_value(); ++l)
        {
            auto it = tokens.begin();
            for (int i = 0; i < static_cast<int>(tokens.size()) + 1 - l; i++, ++it)
            {
                // check if we can merge the sequence (tokens[i], tokens[i+l-1])
                std::string merged =
                    std::accumulate(std::next(it), std::next(it, l), vocab[*it],
                                    [this](auto const &s, auto const &token) { return s + vocab[token]; });

                auto token = strLookUp(merged);

                if (token && (!mergeInfo.has_value() || vocabScores[*token] > mergeInfo->score))
                    // this merge sequence exists in vocab! record its score and position
                    mergeInfo = {vocabScores[*token], it, l, *token};
            }
        }
        if (mergeInfo.has_value())
        {
            *(mergeInfo->start) = mergeInfo->newToken;
            // delete token(s) at position best_idx+1 (and optionally best_idx+2),
            // shift the entire sequence back
            tokens.erase(std::next(mergeInfo->start), std::next(mergeInfo->start, mergeInfo->length));
        }
        else
            break;
    }
}
