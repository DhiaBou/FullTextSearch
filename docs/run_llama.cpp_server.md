# How to setup embeddings server with llama.cpp

## Step 1 - Download and build llama.cpp (CPU build)

Build llama.cpp using `CMake`:

```bash
cmake -B build
cmake --build build --config Release
```

## Step 2 - Download and setup the embeddings model

### Download model

```bash
git clone https://huggingface.co/Snowflake/snowflake-arctic-embed-s
```

### Convert the model to GGUF file format

```bash
# install python deps
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

# convert
python3 convert_hf_to_gguf.py snowflake-arctic-embed-s/ --outfile model-f16.gguf
```

## Step 3 - Start an HTTP Server

```bash
./build/bin/llama-server -m model-f16.gguf --embeddings -c 512 -ngl 99
```
