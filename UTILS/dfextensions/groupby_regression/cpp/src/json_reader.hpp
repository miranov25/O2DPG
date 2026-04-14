// cpp/src/json_reader.hpp
//
// Restricted JSON reader for Phase 13.18.GB fixture schema ONLY.
// NOT a general-purpose JSON library. Parses exactly the 5-key
// top-level structure documented in PHASE_13_18_GB_Fixture_Specification
// v1.0 §4. Unknown top-level keys raise.
//
// Scope-limited per P1-eta (GPT12/GPT13 / Claude23 P2-2).
// Target: ~120 lines total (header + implementation).

#ifndef GBE_JSON_READER_HPP
#define GBE_JSON_READER_HPP

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace gbe {

// Minimal JSON value tree sufficient for the fixture schema.
// Types: null (absent), bool, number (double), string, array, object.
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool        b_val = false;
    double      num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    // Convenience accessors (throw std::invalid_argument on type mismatch).
    bool   as_bool() const;
    double as_number() const;
    int    as_int() const;  // number rounded to int; rejects non-integer
    const std::string& as_string() const;
    const std::vector<JsonValue>& as_array() const;
    const std::map<std::string, JsonValue>& as_object() const;

    // Object field access with type check.
    const JsonValue& at(const std::string& key) const;
    bool contains(const std::string& key) const;
};

// Parse a JSON text into JsonValue. Throws std::invalid_argument with
// a line:column snippet on malformed input.
// Supports the literal "NaN" as a string when it appears as a string
// value; the restricted fixture format uses "NaN" (quoted) to represent
// missing numeric output rather than null (per FIXTURE_SPEC v1.0 §4).
JsonValue parse_json(const std::string& text);

// Read a file and parse it. Convenience wrapper.
JsonValue parse_json_file(const std::string& path);

} // namespace gbe

#endif // GBE_JSON_READER_HPP
