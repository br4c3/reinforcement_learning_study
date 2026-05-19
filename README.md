# reinforcement_learning_study

```bash
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip -o libtorch.zip

unzip libtorch.zip
```

```bash
cmake -S . -B build \
-DCMAKE_PREFIX_PATH=$(pwd)/libtorch

cmake --build build
```

```bash
brew install gnuplot
```

```bash
brew install gnuplot imagemagick
```
