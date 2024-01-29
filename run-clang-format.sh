#!/bin/bash

set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# We are currently standardized on using LLVM/Clang17 for this script.
# If you don't have LLVM17 installed, you can usually install what you need easily via:
#
# sudo apt-get install llvm-17 clang-17 libclang-17-dev clang-tidy-17
# export CLANG_FORMAT_LLVM_INSTALL_DIR=/usr/lib/llvm-17

[ -z "$CLANG_FORMAT_LLVM_INSTALL_DIR" ] && echo "CLANG_FORMAT_LLVM_INSTALL_DIR must point to an LLVM installation dir for this script." && exit
echo CLANG_FORMAT_LLVM_INSTALL_DIR = ${CLANG_FORMAT_LLVM_INSTALL_DIR}

VERSION=$(${CLANG_FORMAT_LLVM_INSTALL_DIR}/bin/clang-format --version)
if [[ ${VERSION} =~ .*version\ 17.* ]]
then
    echo "clang-format version 17 found."
else
    echo "CLANG_FORMAT_LLVM_INSTALL_DIR must point to an LLVM 17 install!"
    exit 1
fi

# Note that we specifically exclude files starting with . in order
# to avoid finding emacs backup files
find "${ROOT_DIR}" \
     \( -name "*.cc" -o -name "*.h" \) -and -not -wholename "*/.*" | \
     xargs ${CLANG_FORMAT_LLVM_INSTALL_DIR}/bin/clang-format -i -style=file
