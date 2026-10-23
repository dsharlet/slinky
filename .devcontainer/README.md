# Development container

An optional, self-contained environment for working on slinky. Nothing in the
CMake build, the Bazel build, or CI uses it; ignore this directory if you
don't need it.

The image is based on Ubuntu 24.04 (like GHA's `ubuntu-latest`) and contains
CMake, Ninja, GCC, Clang, `clang-format`, GDB, and `rsync`.

This is a convenience for developers working on atomic (i.e. rpm-ostree) based
Linux distributions (such as Fedora Atomic), or those who simply want an isolated
development environment.

## CLion

CLion can build, run, and debug inside the image through a Docker/Podman
toolchain:

1. Build the image from the repository root:

   ```sh
   podman build -t slinky-dev -f .devcontainer/Containerfile .devcontainer
   ```

   (Use `docker` instead of `podman` if you prefer.)

2. For Podman, enable the API socket:

   ```sh
   systemctl --user enable --now podman.socket
   ```

3. In **Settings > Build, Execution, Deployment > Docker**, add a Podman (or
   Docker) connection and check that it reports "Connection successful".
4. In **Settings > Build, Execution, Deployment > Toolchains**, add a
   **Docker** toolchain, pick the `slinky-dev` image, and move it to the top
   of the list. CLion should detect CMake, the compiler, and GDB.

With rootless Podman, `root` inside the container maps to your user on the
host, so files created in the build directory are owned by you.

## VS Code and other devcontainer tools

`devcontainer.json` builds the same `Containerfile`, so tools that support the
[dev container spec](https://containers.dev/) can use it directly. For
example, in VS Code run **Dev Containers: Reopen in Container**.

## Command line

From the root of the repository, run:

```sh
podman run --rm -it -v "$PWD":/src -w /src slinky-dev \
    bash -c 'cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build'
```
