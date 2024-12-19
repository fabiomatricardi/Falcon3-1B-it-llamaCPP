<img src='https://github.com/fabiomatricardi/Falcon3-1B-it-llamaCPP/raw/main/falcon3_llamacpp.gif' width=1000>

# Falcon3-1B-it-llamaCPP
How to run Falcon3-1B-instruct with llama.cpp on your pc - this is what we will build

For months I have done the wrong things. But llama.cpp now has pre-compiled binaries at every release. So, for instance, starting from revision b4351 llama.cpp supports also the Falcon3 models.

To be inclusive (all kind of hardware) we will use the binaries for AXV2 support, from release b4358 (the one available at the time of this newsletter writing).

Download the file in your project directory: for me is Falcon3. Create a sub-folder called llamacpp and inside another one called model (we will download the GGF for Falcon3 there).



Unzip all files in the [llama-b4358-bin-win-avx2-x64.zip](https://github.com/ggerganov/llama.cpp/releases/download/b4358/llama-b4358-bin-win-avx2-x64.zip)  archive into the llamacpp directory



Download the   from the MaziyarPanahi Hugging Face repository: I used the Q6 ([Falcon3-1B-Instruct.Q6_K.gguf](https://huggingface.co/MaziyarPanahi/Falcon3-1B-Instruct-GGUF/resolve/main/Falcon3-1B-Instruct.Q6_K.gguf)) quantization, but also the Q8 is good. Save the GGUF file in the subdirectory llamacpp\model.


Open a terminal window in the subdirectory llamacpp, and run
```
.\llama-server.exe -m .\model\Falcon3-1B-Instruct.Q6_K.gguf -c 8192 --port 8001
```

In another terminal with the venv activated run
```
python testFalcon3-1B-it.py
```

