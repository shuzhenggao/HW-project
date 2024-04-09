# A tool for automated prompt optimization for unit test generation on Java.

## Dependency
Our tool requires the Java environment and the installation of Defects4J.


Requirements
----------------
 - Java 1.8
 - Git >= 1.9
 - SVN >= 1.8
 - Perl >= 5.0.12

All required Perl modules are listed here.

```bash
requires 'DBI',         '>= 1.63';
requires 'DBD::CSV',    '>= 0.48';
requires 'URI',         '>= 1.72';
requires 'JSON',        '>= 2.97';
requires 'JSON::Parse', '>= 0.55';
requires 'List::Util',  '>= 1.33';
```

On many Unix platforms, these required Perl modules are installed by default.
If this is not the case, see the instructions below for how to install them.

Steps to set up Defects4J
----------------

1. Clone Defects4J:
    - `git clone https://github.com/rjust/defects4j`

2. Initialize Defects4J (download the project repositories and external libraries, which are not included in the git repository for size purposes and to avoid redundancies):
   If you do not have `cpanm` installed, use cpan or a cpan wrapper to install the perl modules listed in `cpanfile`.
    - `cd defects4j`
    - `cpanm --installdeps .`
    - `./init.sh`

3. Add Defects4J's executables to your PATH:
    - `export PATH=$PATH:"path2defects4j"/framework/bin`

4. Check installation:
    - `defects4j info -p Lang`

On some platforms such as Windows, you might need to use `perl "fullpath"\defects4j`
where these instructions say to use `defects4j`.



## Usage
If you want to use LLMs of OpenAI and Google by API service. Please first obtain their API keys.

It is recommended to use the API service for seed prompt generation.

Fill the config.yaml
----------------

- seed_number: number of seed prompts.

- model_name: name of used LLMs. For models of OpenAI, please select from one model the [doc of OpenAI](https://platform.openai.com/docs/models/overview). For Gemini, please select `gemini-pro`. For open-source models, please select from [Hugging Face](https://huggingface.co/) or input the address in your own machine.

- model_api: the API key of your LLM if you use models of OpenAI and Gemini. 

- iteration_number: number of iterations.

- max_test_cases: maximum number of generated cases for one focal method.

- generated_number: number of newly generated prompts in each iteration.

- seed_prompt_addr: address of seed prompt. You can manually write some seed prompts for yourself.


Run auto_prompt.py
----------------
An example of using LLMs:
```python
python auto_prompt.py --seed_number 5 --iteration_number 5 --max_test_cases 10 --generated_number 2 --seed_prompt_addr seed_prompt.txt --model_name gpt-3.5-turbo-0125 --model_api sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

An example of using open source models (CodeLlama):
```python
python auto_prompt.py --seed_number 5 --iteration_number 5 --max_test_cases 10 --generated_number 2 --seed_prompt_addr seed_prompt.txt --model_name CodeLlama-34b-Instruct-hf --model_api sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

An example of the content in seed_prompt.txt:
```
Generate comprehensive unit tests for the provided Java methods, ensuring they cover both regular operation and edge cases. Test names should be descriptive, reflecting the scenarios being tested, and assertions must confirm that the method's output matches the expected results. Include tests for the examples outlined in the method's documentation to ensure accurate validation of functionality.
You will be provided with the focal method, its import statements, and the class signature. Your responsibility is to create a maximum of 10 test cases for this focal method to guarantee comprehensive test coverage. \nPlease provide the necessary details for the focal method, such as its name, input parameters, expected output, and any additional context. With this information, I will generate the corresponding test cases to enhance the coverage of your focal method.
```
