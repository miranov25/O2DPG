// cpp/src/json_reader.cpp — restricted JSON parser for fixture schema.
#include "json_reader.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gbe {

// Accessors --------------------------------------------------------------

bool JsonValue::as_bool() const {
    if (type != Type::Bool) throw std::invalid_argument("expected bool");
    return b_val;
}
double JsonValue::as_number() const {
    if (type != Type::Number) throw std::invalid_argument("expected number");
    return num_val;
}
int JsonValue::as_int() const {
    const double d = as_number();
    const int r = static_cast<int>(d);
    if (static_cast<double>(r) != d) {
        throw std::invalid_argument("expected integer-valued number");
    }
    return r;
}
const std::string& JsonValue::as_string() const {
    if (type != Type::String) throw std::invalid_argument("expected string");
    return str_val;
}
const std::vector<JsonValue>& JsonValue::as_array() const {
    if (type != Type::Array) throw std::invalid_argument("expected array");
    return arr_val;
}
const std::map<std::string, JsonValue>& JsonValue::as_object() const {
    if (type != Type::Object) throw std::invalid_argument("expected object");
    return obj_val;
}
const JsonValue& JsonValue::at(const std::string& key) const {
    const auto& obj = as_object();
    auto it = obj.find(key);
    if (it == obj.end()) {
        throw std::invalid_argument("missing key: " + key);
    }
    return it->second;
}
bool JsonValue::contains(const std::string& key) const {
    if (type != Type::Object) return false;
    return obj_val.find(key) != obj_val.end();
}

// Parser -----------------------------------------------------------------

namespace {

struct Parser {
    const std::string& text;
    std::size_t pos = 0;

    explicit Parser(const std::string& t) : text(t) {}

    [[noreturn]] void err(const std::string& what) const {
        // Compute line/column for error message
        std::size_t line = 1, col = 1;
        for (std::size_t i = 0; i < pos && i < text.size(); ++i) {
            if (text[i] == '\n') { ++line; col = 1; } else { ++col; }
        }
        std::ostringstream oss;
        oss << "JSON parse error at " << line << ":" << col << " — " << what;
        throw std::invalid_argument(oss.str());
    }

    void skip_ws() {
        while (pos < text.size()) {
            const char c = text[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos;
            else break;
        }
    }

    char peek() {
        if (pos >= text.size()) err("unexpected EOF");
        return text[pos];
    }

    char eat() {
        if (pos >= text.size()) err("unexpected EOF");
        return text[pos++];
    }

    bool try_eat(char c) {
        skip_ws();
        if (pos < text.size() && text[pos] == c) { ++pos; return true; }
        return false;
    }

    void expect(char c) {
        skip_ws();
        if (pos >= text.size() || text[pos] != c) {
            std::ostringstream oss;
            oss << "expected '" << c << "'";
            err(oss.str());
        }
        ++pos;
    }

    JsonValue parse_value();
    JsonValue parse_object();
    JsonValue parse_array();
    JsonValue parse_string();
    JsonValue parse_number();
};

JsonValue Parser::parse_value() {
    skip_ws();
    if (pos >= text.size()) err("unexpected EOF");
    const char c = text[pos];
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return parse_string();
    if (c == 't' || c == 'f') {
        // true / false
        if (text.compare(pos, 4, "true") == 0) {
            pos += 4; JsonValue v; v.type = JsonValue::Type::Bool;
            v.b_val = true; return v;
        }
        if (text.compare(pos, 5, "false") == 0) {
            pos += 5; JsonValue v; v.type = JsonValue::Type::Bool;
            v.b_val = false; return v;
        }
        err("expected true/false");
    }
    if (c == 'n') {
        if (text.compare(pos, 4, "null") == 0) {
            pos += 4; JsonValue v; v.type = JsonValue::Type::Null;
            return v;
        }
        err("expected null");
    }
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    err("unexpected character");
}

JsonValue Parser::parse_object() {
    JsonValue v; v.type = JsonValue::Type::Object;
    expect('{');
    skip_ws();
    if (try_eat('}')) return v;
    while (true) {
        skip_ws();
        JsonValue k = parse_string();
        expect(':');
        v.obj_val.emplace(k.str_val, parse_value());
        skip_ws();
        if (try_eat(',')) continue;
        expect('}');
        break;
    }
    return v;
}

JsonValue Parser::parse_array() {
    JsonValue v; v.type = JsonValue::Type::Array;
    expect('[');
    skip_ws();
    if (try_eat(']')) return v;
    while (true) {
        v.arr_val.push_back(parse_value());
        skip_ws();
        if (try_eat(',')) continue;
        expect(']');
        break;
    }
    return v;
}

JsonValue Parser::parse_string() {
    JsonValue v; v.type = JsonValue::Type::String;
    expect('"');
    while (pos < text.size()) {
        const char c = text[pos++];
        if (c == '"') return v;
        if (c == '\\') {
            if (pos >= text.size()) err("bad escape");
            const char e = text[pos++];
            switch (e) {
                case '"': v.str_val.push_back('"'); break;
                case '\\': v.str_val.push_back('\\'); break;
                case '/': v.str_val.push_back('/'); break;
                case 'b': v.str_val.push_back('\b'); break;
                case 'f': v.str_val.push_back('\f'); break;
                case 'n': v.str_val.push_back('\n'); break;
                case 'r': v.str_val.push_back('\r'); break;
                case 't': v.str_val.push_back('\t'); break;
                // \uXXXX not supported — fixture schema does not use it
                default: err("unsupported escape");
            }
        } else {
            v.str_val.push_back(c);
        }
    }
    err("unterminated string");
}

JsonValue Parser::parse_number() {
    JsonValue v; v.type = JsonValue::Type::Number;
    std::size_t start = pos;
    if (text[pos] == '-') ++pos;
    while (pos < text.size()) {
        const char c = text[pos];
        if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E'
            || c == '+' || c == '-') {
            ++pos;
        } else break;
    }
    const std::string num = text.substr(start, pos - start);
    char* endp = nullptr;
    v.num_val = std::strtod(num.c_str(), &endp);
    if (endp == num.c_str()) err("malformed number");
    return v;
}

} // anonymous namespace

JsonValue parse_json(const std::string& text) {
    Parser p(text);
    JsonValue v = p.parse_value();
    p.skip_ws();
    if (p.pos != text.size()) {
        p.err("trailing characters after top-level value");
    }
    return v;
}

JsonValue parse_json_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::invalid_argument("cannot open file: " + path);
    }
    std::ostringstream oss;
    oss << f.rdbuf();
    return parse_json(oss.str());
}

} // namespace gbe
