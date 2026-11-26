#!/bin/sh
# Fail if the engine crate mentions a floating-point type or contains a float literal.
# Comments, (single-line) string literals and hexadecimal literals are ignored.
status=0
for file in $(find crates/detllm -name '*.rs'); do
    if sed -e 's|//.*$||' -e 's/"[^"]*"//g' -e 's/0x[0-9a-fA-F_]*//g' "$file" \
        | grep -nE 'f(32|64)|[0-9]\.[0-9]|[0-9]e[-+]?[0-9]'; then
        echo "floating point found in $file" >&2
        status=1
    fi
done
exit $status
