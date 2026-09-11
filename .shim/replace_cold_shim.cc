// Prebuilt ONNX Runtime binaries (pyke CDN, built against libstdc++ 13/14)
// reference std::__cxx11::basic_string<char>::_M_replace_cold, which system
// libstdc++ 12 does not export. Semantics: replace [pos, pos+len1) with
// s[0..len2) where p points at data()+pos (non-aliased, "cold" variant of
// _M_replace). Forwarding to the public replace() is always correct.
#include <string>
using std::string;
extern "C" __attribute__((visibility("default"), used))
void _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE15_M_replace_coldEPcmPKcmm(
        void* self, char* p, unsigned long len1, const char* s, unsigned long len2) {
    string* str = static_cast<string*>(self);
    unsigned long pos = static_cast<unsigned long>(p - str->data());
    str->replace(pos, len1, s, len2);
}
extern "C" __attribute__((visibility("default"), used))
void _ZNSt7__cxx1112basic_stringIwSt11char_traitsIwESaIwEE15_M_replace_coldEPwmPKwmm(
        void* self, wchar_t* p, unsigned long len1, const wchar_t* s, unsigned long len2) {
    std::wstring* str = static_cast<std::wstring*>(self);
    unsigned long pos = static_cast<unsigned long>(p - str->data());
    str->replace(pos, len1, s, len2);
}
