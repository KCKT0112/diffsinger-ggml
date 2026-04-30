#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dsp {

struct DSSegment {
    double offset = 0.0;                    // seconds, for stitching

    std::vector<std::string> ph_names;      // [L], raw phoneme names from .ds
    std::vector<int32_t> tokens;
    std::vector<int32_t> ph2word;           // [L], 1-based word index
    std::vector<float>   word_dur;          // [W], frame counts per word
    std::vector<float>   ph_dur;            // [L], frame counts per phone
    std::vector<int32_t> mel2ph;            // [T], 1-based phone index
    std::vector<float>   f0_hz;             // [T]
    std::vector<float>   pitch_midi;        // [T]
    std::vector<int32_t> languages;         // [L], all zeros if no lang_id

    // Note-level info (for pitch model)
    std::vector<float>   note_midi;         // [N], MIDI pitch (-1 = rest/pad)
    std::vector<int32_t> note_rest;         // [N], 1 if rest
    std::vector<int32_t> note_dur_frames;   // [N], duration in frames
    std::vector<int32_t> mel2note;          // [T], 1-based note index
    std::vector<float>   base_pitch;        // [T], MIDI semitones from note expansion

    // Variance curves (empty if not present in .ds → will be predicted)
    std::vector<float>   breathiness;       // [T]
    std::vector<float>   voicing;           // [T]
    std::vector<float>   tension;           // [T]
    std::vector<float>   energy;            // [T]
    std::vector<float>   gender;            // [T] (key_shift)
    std::vector<float>   velocity;          // [T] (speed)
};

// Load phoneme-id map from phonemes.json.
// `default_lang` is used to expand "lang/phone" → "phone" entries for the given language.
// Returns empty map on failure.
std::unordered_map<std::string, int> load_phoneme_map(const std::string & path,
                                                      const std::string & default_lang = "zh");

// Load speaker name→id map from spk_map.json.
std::unordered_map<std::string, int> load_spk_map(const std::string & path);

// Parse a .ds file into segments.
// sr/hop are the model's sample rate and hop size (typically 44100/512).
// Returns empty vector on failure.
std::vector<DSSegment> parse_ds_file(const std::string & ds_path,
                                     const std::unordered_map<std::string, int> & phoneme_map,
                                     int sr, int hop);

} // namespace dsp
