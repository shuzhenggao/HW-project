import os
import re
import regex
import openai
import shutil
import argparse
import subprocess
import multiprocessing
from tqdm import tqdm
from time import sleep
import numpy as np
import torch
import jsonlines
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import tarfile
import jsonlines
import xml.etree.ElementTree as ET
import google.generativeai as genai


llm_api_key = ''

error_file = open("stderr.txt", "wb")

fix_regression_file = '''
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({ RegressionTest0.class})
public class RegressionTest {
}
'''

fix_compile_file = '''
import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.junit.FixMethodOrder;
import org.junit.runners.MethodSorters;

'''
fix_test_file = '''
import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.junit.FixMethodOrder;
import org.junit.runners.MethodSorters;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class RegressionTest0 {

    public static boolean debug = false;
'''


def build_d4j_train():
    bugs = ['Lang-1', 'Math-5', 'Time-4']
    # # Chart:
    # for i in range(1, 27):
    #     bugs.append('Chart-{}'.format(i))
    # # Closure
    # for i in range(1, 134):
    #     if i != 63 and i != 93:
    #         bugs.append('Closure-{}'.format(i))
    # # Lang
    # for i in range(1, 66):
    #     if i != 2:
    #         bugs.append('Lang-{}'.format(i))
    # # Math
    # for i in range(1, 107):
    #     bugs.append('Math-{}'.format(i))
    # # Mockito
    # for i in range(1, 39):
    #     bugs.append('Mockito-{}'.format(i))
    # # Time
    # for i in range(1, 28):
    #     bugs.append('Time-{}'.format(i))
    return bugs

def build_d4j_test():
    bugs = ['Chart-1', 'Chart-3', 'Lang-3', 'Math-2', 'Time-7']
    return bugs


def get_loc_file(bug_id):
    dirname = os.path.dirname(__file__)
    loc_file = './location/groundtruth/%s/%s' % (bug_id.split("-")[0].lower(), bug_id.split("-")[1])
    loc_file = os.path.join(dirname, loc_file)
    if os.path.isfile(loc_file):
        return loc_file
    else:
        print(loc_file)
        return ""
    
def get_location(bug_id):
    source_dir = os.popen("defects4j export -p dir.src.classes -w ./tmp/" + bug_id).readlines()[-1].strip() + "/"
    location = []
    loc_file = get_loc_file(bug_id)
    if loc_file == "":
        return location
    lines = open(loc_file, 'r').readlines()
    for loc_line in lines[:1]:
        loc_line = loc_line.split("||")[0]
        classname, line_id = loc_line.split(':')
        classname = ".".join(classname.split(".")[:-1])
        if '$' in classname:
            classname = classname[:classname.index('$')]
        file = source_dir + "/".join(classname.split(".")) + ".java"
        location.append((file, int(line_id) - 1))

    return location[0]


def model_inference(cfg, chunk_data):
    data = chunk_data['data']
    output_path = chunk_data['output_path']

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = "<pad>"
    tokenizer.padding_side = "left"
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        return_dict=True,
        torch_dtype=torch.float32,
    ).to(torch.device(device))
    model.to(torch.device(device))
    existing_prediction=set()
    if os.path.exists(os.path.join(output_path, 'prediction.jsonl')):
        with jsonlines.open(os.path.join(output_path, 'prediction.jsonl')) as f:
            for obj in f:
                existing_prediction.add(obj['prompt'].strip()+'\n'+obj['method'].strip())
    for idx in tqdm(range(len(data))):
        prompt = data[idx]['prompt'].strip()+'\n'+data[idx]['method'].strip()
        if prompt in existing_prediction:
            continue
        messages = '<s>[INST] <<SYS>>\n{{You are a software developer and now you will help to write unit cases. Please follow the instructions and reply with the unit cases in code blocks (```java ```). Please do not reply with over '+str(cfg.max_test_cases)+' test cases.}}\n<</SYS>>\n\n{{'+prompt+'}}[/INST]'
        inputs_ft = tokenizer(messages, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        max_length = 2048 + inputs_ft.input_ids.flatten().size(0)
        if max_length > 4096:
            print("warning: max_length {} is greater than the context window {}".format(max_length, 4096))
            continue
        generated_ids = model.generate(**inputs_ft, max_new_tokens=2048, temperature=0.0)
        prediction = tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=False)[0]
        answer = {'prompt':data[idx]['prompt'], 'method':data[idx]['method'], 'ut':post_process(prediction.strip(), idx), 'prediction':prediction.strip()}
        with jsonlines.open(os.path.join(output_path, 'prediction.jsonl'), 'a') as f:
            f.write_all([answer])




def api_official(cfg, chunk_data):
    data = chunk_data['data']
    output_path = chunk_data['output_path']
    if cfg.model_name == 'gemini-pro':
        genai.configure(api_key=llm_api_key)
        generation_config = { 
            "temperature": 0.7,
            "candidate_count": 1, 
        }
        safety_settings = [ 
            {
                "category": "HARM_CATEGORY_HARASSMENT", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
            }, 
            { 
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
            }, 
            { 
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
            }, 
            { 
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
            } 
        ] 
        model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config, safety_settings=safety_settings) 
        for idx in tqdm(range(len(data))):
            prompt = data[idx]['prompt']+data[idx]['method']
            if prompt in existing_prediction:
                continue
            success = 0
            fail_count = 0
            while success!=1:
                messages = [
                    {"role": "user", "parts": ["You are a software developer and now you will help to write unit tests. Please follow the instructions and reply with the unit cases in code blocks (```java ```). Please do not reply with over {} test cases.".format(cfg.max_test_cases)]},
                    {'role':'model','parts':['OK, I will help you write unit test.']},
                    {"role": "user", "parts": [prompt]},
                ]
                try:
                    response = model.generate_content(messages)
                    success=1
                    answer = {'prompt':data[idx]['prompt'], 'method':data[idx]['method'], 'ut':post_process(response.text, idx), 'prediction':response.text}
                    with jsonlines.open(os.path.join(output_path, 'prediction.jsonl'), 'a') as f:
                        f.write_all([answer])
                    sleep(2)
                except Exception  as e:
                    info = e.args[0]
                    fail_count+=1
                    if 'Max retries exceeded with url:' in info:
                        sleep(2*fail_count)
                    print(info)
                if fail_count>10:
                    print('{} fail more than 10 times'.format(str(fail_count)))
                    break
    else:
        model = cfg.model_name
        openai.api_key = llm_api_key
        existing_prediction=set()
        if os.path.exists(os.path.join(output_path, 'prediction.jsonl')):
            with jsonlines.open(os.path.join(output_path, 'prediction.jsonl')) as f:
                for obj in f:
                    existing_prediction.add(obj['prompt'].strip()+'\n'+obj['method'].strip())
        for idx in tqdm(range(len(data))):
            prompt = data[idx]['prompt'].strip()+'\n'+data[idx]['method'].strip()
            if prompt in existing_prediction:
                continue
            success = 0
            fail_count = 0
            while success!=1:
                messages = [
                        {"role": "system", "content": "You are a software developer and now you will help to write unit cases. Please follow the instructions and reply with the unit cases in code blocks (```java ```). Please do not reply with over {} test cases.".format(cfg.max_test_cases)},
                        {"role": "user", "content": prompt},
                        ]
                try:
                    model = cfg.model_name
                    response = openai.ChatCompletion.create(model=model, messages=messages, n=1, temperature=0)
                    success=1
                    answer = {'prompt':data[idx]['prompt'], 'method':data[idx]['method'], 'ut':post_process(response["choices"][0]['message']['content'].strip(), idx), 'prediction':response["choices"][0]['message']['content'].strip()}
                    with jsonlines.open(os.path.join(output_path, 'prediction.jsonl'), 'a') as f:
                        f.write_all([answer])
                    sleep(2)
                except Exception  as e:
                    info = e.args[0]
                    fail_count+=1
                    if 'Max retries exceeded with url:' in info:
                        sleep(2*fail_count)
                    print(info)
                if fail_count>10:
                    print('{} fail more than 10 times'.format(str(fail_count)))
                    break


def post_process(string, idx):
    string.replace('``` java', '```java')
    string.replace('```Java', '```java')
    string.replace('``` Java', '```java')
    methods = string.split('```java')
    ut = []
    for method in methods:
        if '```' in method:
            ut.append(method.split('```')[0])
    return '\n'.join(ut)


def ut_generation(cfg, bug_ids, prompt):
    for bug_id in bug_ids:
        print('Generating for {}'.format(bug_id))
        subprocess.run('rm -rf ' + './tmp/' + bug_id, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
            bug_id.split('-')[0], bug_id.split('-')[1] + 'b', ('./tmp/' + bug_id)), stdout=subprocess.PIPE, stderr=error_file, shell=True)
        location = get_location(bug_id)
        file = './tmp/' + bug_id + '/' + location[0]
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.readlines()
            code = '\n'.join(code)
        class_name = location[0].split('.java')[0].split('/')[-1]
        methods = extract_methods(code, class_name)
        data = []
        if not os.path.exists('./output/' + bug_id):
            os.makedirs('./output/' + bug_id)
        for method in methods:
            data.append({'prompt':prompt, 'method':'import statements:\n{}class signature:\n{}\nfocal method:\n{}\n'.format(method['import'], method['signature'].strip(), method['method'])})
        tmp_data={}
        tmp_data['data'] = data
        tmp_data['output_path'] = './output/' + bug_id
        tmp_data['api'] = 0
        if cfg.model_name: 
            api_official(cfg, tmp_data)
        else:
            model_inference(cfg, tmp_data)



def add_suffix_to_test_methods(java_code, start_id):
    uts = java_code.split('@Test')
    processed_ut = []
    for idx in range(len(uts)):
        ut = uts[idx]
        if '(' not in ut:
            continue
        if not ut.strip().startswith('public'):
            continue
        left_parenthesis_index = ut.index('(')
        start_of_function_name = ut.rfind(' ', 0, left_parenthesis_index) + 1
        before_function_name = ut[:start_of_function_name]
        function_name = ut[start_of_function_name:left_parenthesis_index]
        after_function_name = ut[left_parenthesis_index:]
        updated_ut = before_function_name + function_name + str(idx+start_id) + after_function_name
        processed_ut.append('@Test'+updated_ut+'\n')
    return processed_ut, idx+start_id

        

def extract_methods(code, class_name):
    pattern = r'^package\s+[\w\.]+;'
    package_statements = re.findall(pattern, code, re.MULTILINE)[0].strip()
    import_pattern = re.compile(
        r'^import\s+[\w.]+;$', 
        re.MULTILINE
    )
    imports = import_pattern.findall(code)
    import_statements = package_statements+'\n'
    for i in imports:
        import_statements+=(i+'\n')
    
    class_header_regex = r"public\s+(?:[\w\s]+)?class\s+[\w<>,\s]+{"
    match = re.search(class_header_regex, code)
    class_header = (match.group(0) if match else None)+'\n'

    def extract_function(java_code):
        methods = []
        index = 0
        while index < len(java_code):
            method_start = java_code.find('public ', index)
            if method_start == -1:
                break  
            
            left_brace = java_code.find('{', method_start)
            class_index = java_code.find('class', method_start)
            comment_index = java_code.find('*/', method_start)
            semicolon = java_code.find(';', method_start)
            if semicolon != -1 and (semicolon < left_brace or left_brace == -1):
                index = semicolon + 1
                continue
            elif class_index != -1 and class_index < left_brace:
                index = class_index + 1
                continue
            elif comment_index != -1 and comment_index < left_brace:
                index = comment_index + 1
                continue
            if left_brace == -1:
                break

            braces_count = 1
            index = left_brace + 1
            while index < len(java_code) and braces_count > 0:
                char = java_code[index]
                if char == '{':
                    braces_count += 1
                elif char == '}':
                    braces_count -= 1
                index += 1
            if braces_count == 0:
                method_end = index
                methods.append(java_code[method_start:method_end])
    
        return methods
    
    matches = extract_function(code)
    methods = []
    for function in matches:
        if class_name in function and function.split(class_name)[1].strip()[0]=='(':
            class_header+=function+'\n'
        else:
            methods.append({'import':import_statements, 'method':function})
    for method in methods:
        method['signature'] = class_header
    
    return methods



def evaluate_coverage(bug_ids, prompt):
    line_score=0
    condition_score=0
    start_id=0
    for bug_id in bug_ids:
        subprocess.run('rm -rf ' + './tmp/' + bug_id, stdout=subprocess.PIPE, stderr=error_file, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
            bug_id.split('-')[0], bug_id.split('-')[1] + 'b', ('./tmp/' + bug_id)), stdout=subprocess.PIPE, stderr=error_file, shell=True)
        location = get_location(bug_id)
        file = './tmp/' + bug_id + '/' + location[0]
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.readlines()
            code = '\n'.join(code)

        # package info
        pattern = r'^package\s+[\w\.]+;'
        package_statements = re.findall(pattern, code, re.MULTILINE)[0].strip()
        package = package_statements.split('package')[1].split(';')[0].strip()
        package_dir = '/'+package.replace('.', '/')
        
        # class info
        class_name = location[0].split('.java')[0].split('/')[-1]

        # import info
        import_pattern = re.compile(
            r'^import\s+[\w.]+;$', 
            re.MULTILINE
        )
        imports = import_pattern.findall(code)
        import_statements = package_statements.replace('packge', 'import')+'\n'
        import_statements += 'import '+package_statements.split('package')[1].split(';')[0]+'.'+class_name+';\n'
        for i in imports:
            import_statements+=(i+'\n')

        # write ut file
        existing_methods = ['import statements:\n{}class signature:\n{}\nfocal method:\n{}\n'.format(item['import'], item['signature'].strip(), item['method']) for item in extract_methods(code, class_name)]
        predictions=''
        with jsonlines.open(os.path.join('./output/' + bug_id, 'prediction.jsonl')) as f:
            for obj in f:
                if obj['prompt'].strip()==prompt.strip() and obj['method'] in existing_methods:
                    predictions+=(obj['ut']+'\n')

        # test compile
        processed_ut,  start_id = add_suffix_to_test_methods(predictions, start_id)
        fail_count=0
        succuess_ut = []
        match = False
        if os.path.exists('./output/'+bug_id+'/filtered_prediction'):
            with jsonlines.open('./output/'+bug_id+'/filtered_prediction', 'r') as f:
                for obj in f:
                    if obj['prompt'].strip() == prompt.strip():
                        succuess_ut = obj['succuess_ut']
                        match = True
        if not match:
            for ut in tqdm(processed_ut):
                ut_file = import_statements+fix_test_file+ut+'}'
                regression_file = package_statements+fix_regression_file
                if os.path.exists('./coverage_test/'+bug_id):
                    shutil.rmtree('./coverage_test/'+bug_id)
                os.makedirs('./coverage_test/'+bug_id+package_dir)
                with open(os.path.join('./coverage_test/'+bug_id+package_dir, 'RegressionTest.java'), 'w') as f:
                    f.write(regression_file)
                with open(os.path.join('./coverage_test/'+bug_id+package_dir, 'RegressionTest0.java'), 'w') as f:
                    f.write(ut_file)
                    
                with tarfile.open('./coverage_test/ut.tar.bz2', "w:bz2") as tar:
                    original_directory = os.getcwd()
                    os.chdir('./coverage_test/'+bug_id)
                    for item in os.listdir('.'):
                        tar.add(item, arcname=item)
                    os.chdir(original_directory)
                subprocess.run('rm -rf ' + './tmp/' + bug_id, stdout=subprocess.PIPE, stderr=error_file, shell=True)
                subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
                    bug_id.split('-')[0], bug_id.split('-')[1] + 'b', ('./tmp/' + bug_id)), stdout=subprocess.PIPE, stderr=error_file, shell=True)
                # result = subprocess.run("defects4j coverage -s ./coverage_test/ut.tar.bz2 -w %s" % (('./tmp/' + bug_id)), 
                #                         stdout=subprocess.PIPE, stderr=error_file, shell=True)
                result = subprocess.run("defects4j coverage -s ./coverage_test/ut.tar.bz2 -w %s" % (('./tmp/' + bug_id)), 
                                        stdout=subprocess.PIPE, stderr=error_file, shell=True)
                result_str = result.stdout.decode('utf-8')
                if 'coverage' in result_str:
                    with open("stderr.txt", 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                        if 'Some tests failed' in code or 'Cannot compile extracted test suite' in code :
                            fail_count+=1
                        else:
                            succuess_ut.append(ut)
            with jsonlines.open('./output/'+bug_id+'/filtered_prediction', 'a') as f:
                f.write_all([{'prompt':prompt, 'succuess_ut':succuess_ut}])


        # test coverage
        print('Processing ', str(bug_id))
        print('Processed_ut:{} Succuess_ut:{} Fail ut:{}'.format(len(processed_ut), len(succuess_ut), fail_count))
        ut_file = import_statements+fix_test_file+'\n'.join(succuess_ut)+'}'
        regression_file = package_statements+fix_regression_file
        if os.path.exists('./coverage_test/'+bug_id):
            shutil.rmtree('./coverage_test/'+bug_id)
        os.makedirs('./coverage_test/'+bug_id+package_dir)
        with open(os.path.join('./coverage_test/'+bug_id+package_dir, 'RegressionTest.java'), 'w') as f:
            f.write(regression_file)
        with open(os.path.join('./coverage_test/'+bug_id+package_dir, 'RegressionTest0.java'), 'w') as f:
            f.write(ut_file)
            
        with tarfile.open('./coverage_test/ut.tar.bz2', "w:bz2") as tar:
            original_directory = os.getcwd()
            os.chdir('./coverage_test/'+bug_id)
            for item in os.listdir('.'):
                tar.add(item, arcname=item)
            os.chdir(original_directory)
        subprocess.run('rm -rf ' + './tmp/' + bug_id, stdout=subprocess.PIPE, stderr=error_file, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
            bug_id.split('-')[0], bug_id.split('-')[1] + 'b', ('./tmp/' + bug_id)), stdout=subprocess.PIPE, stderr=error_file, shell=True)
        result = subprocess.run("defects4j coverage -s ./coverage_test/ut.tar.bz2 -w %s" % (('./tmp/' + bug_id)), 
                                stdout=subprocess.PIPE, stderr=error_file, shell=True)
        result_str = result.stdout.decode('utf-8')
        line_coverage_match = re.search(r'Line coverage:\s*(\d+\.\d+)%', result_str)
        condition_coverage_match = re.search(r'Condition coverage:\s*(\d+\.\d+)%', result_str)
        line_coverage = float(line_coverage_match.group(1)) if line_coverage_match else None
        condition_coverage = float(condition_coverage_match.group(1)) if condition_coverage_match else None
        print('line coverage: {}, condition coverage: {}'.format(line_coverage, condition_coverage))
        
        # test hit buggy line
        tree = ET.parse('./tmp/'+bug_id+'/coverage.xml')
        root = tree.getroot()
        lines_info = {}
        for line in root.findall(".//line"):
            line_number = line.get('number')
            hits = line.get('hits')
            if line_number is not None and hits is not None:
                lines_info[int(line_number)] = int(hits)
        if location[1] not in lines_info:
            hit=-1
        elif lines_info[location[1]] >0:
            hit=1
        else:
            hit=0
        print(f"Hit: {hit}")

        # test bug trigger
        subprocess.run('rm -rf ' + './coverage_test/detection/*', stdout=subprocess.PIPE, stderr=error_file, shell=True)
        subprocess.run('mv ./coverage_test/ut.tar.bz2 ./coverage_test/detection/{}f-llm.1.tar.bz2'.format(bug_id), stdout=subprocess.PIPE, stderr=error_file, shell=True)
        subprocess.run("run_bug_detection.pl -p {} -d /home/ubuntu/UTgen/coverage_test/detection -o ./ -t /home/ubuntu/tmp".format(bug_id.split('-')[0]), stdout=subprocess.PIPE, stderr=error_file, shell=True)
        with open('bug_detection', 'r', encoding='utf-8', errors='ignore') as f:
            code = f.readlines()
        number = code[-1].split(',')[-1]
        print(f"Num_trigger: {number}")

        line_score += line_coverage
        condition_score += condition_coverage
    return line_score/len(bug_ids), condition_score/len(bug_ids)


def prompt_improvement(cfg, selected_prompts):
    openai.api_key = llm_api_key
    model = cfg.model_name
    success = 0
    fail_count = 0
    while success!=1:
        messages = [
                {"role": "system", "content": "You are a prompt engineering and now you will help to write prompt. Make the prompt you write in block (``` ```)."},
                {"role": "user", "content": 'Cross over the following instructions and generate a new instruction. Instruction 1:{}\n Instruction 2:{}\n Instruction 3:{}'.format(selected_prompts[0], selected_prompts[1], selected_prompts[2])}
                ]
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, n=cfg.generated_number, temperature=0.7)
            answers = []
            for answer in response["choices"]:
                answers.append(answer['message']['content'].strip().split('```')[1:-1])
            success=1
        except Exception  as e:
            info = e.args[0]
            fail_count+=1
            if 'Max retries exceeded with url:' in info:
                sleep(2*fail_count)
            print(info)
        if fail_count>10:
            print('{} fail more than 10 times'.format(str(fail_count)))
            break
    return selected_prompts[cfg.seed_number-cfg.generated_number]+answers


def generate_seed_prompt(cfg, prompts):
    openai.api_key = llm_api_key
    model = cfg.model_name
    success = 0
    fail_count = 0
    input_output_pair = '''Input:
Focal method:
/**
 * <p>Convert a <code>String</code> to a <code>double</code>, returning
 * <code>0.0d</code> if the conversion fails.</p>
 *
 * <p>If the string <code>str</code> is <code>null</code>,
 * <code>0.0d</code> is returned.</p>
 *
 * <pre>
 *   NumberUtils.toDouble(null)   = 0.0d
 *   NumberUtils.toDouble("")     = 0.0d
 *   NumberUtils.toDouble("1.5")  = 1.5d
 * </pre>
 *
 * @param str the string to convert, may be <code>null</code>
 * @return the double represented by the string, or <code>0.0d</code>
 *  if conversion fails
 * @since 2.1
 */
public static double toDouble(final String str) {
    return toDouble(str, 0.0d);
}

Import statements:
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.apache.commons.lang3.StringUtils;


Class signature:
public class NumberUtils {
    public NumberUtils() {
        super();
    }


Output:
```java
@Test
public void test0001() throws Throwable {
    double double1 = org.apache.commons.lang3.math.NumberUtils.toDouble("hi!");
    assertTrue("'" + double1 + "' != '" + 0.0d + "'", double1 == 0.0d);
}```


Input:
Focal method:
/**
 * Return the absolute value of this complex number.
 * Returns {@code NaN} if either real or imaginary part is {@code NaN}
 * and {@code Double.POSITIVE_INFINITY} if neither part is {@code NaN},
 * but at least one part is infinite.
 *
 * @return the absolute value.
 */
public double abs() {
    if (isNaN) {
        return Double.NaN;
    }
    if (isInfinite()) {
        return Double.POSITIVE_INFINITY;
    }
    if (FastMath.abs(real) < FastMath.abs(imaginary)) {
        if (imaginary == 0.0) {
            return FastMath.abs(real);
        }
        double q = real / imaginary;
        return FastMath.abs(imaginary) * FastMath.sqrt(1 + q * q);
    } else {
        if (real == 0.0) {
            return FastMath.abs(imaginary);
        }
        double q = imaginary / real;
        return FastMath.abs(real) * FastMath.sqrt(1 + q * q);
    }
}

Import statements:
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.FieldElement;
import org.apache.commons.math3.exception.NotPositiveException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathUtils;



Class signature:
public class Complex implements FieldElement<Complex>, Serializable  {
    public Complex(double real) {
        this(real, 0.0);
    }
    public Complex(double real, double imaginary) {
        this.real = real;
        this.imaginary = imaginary;

        isNaN = Double.isNaN(real) || Double.isNaN(imaginary);
        isInfinite = !isNaN &&
            (Double.isInfinite(real) || Double.isInfinite(imaginary));
    }


Output:
```java
@Test
public void testAbsRealZeroImaginaryPositive300() {
    Complex complex = new Complex(0.0, 4.0);
    assertEquals(4.0, complex.abs(), 0.0001);
}```'''
    while success!=1:
        messages = [
                {"role": "system", "content": "You are a prompt engineering and now you will help to write prompt. Make the prompt you write in block (``` ```) and donot explain anything."},
                {"role": "user", "content": 'I gave a friend an instruction and two inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs: '+input_output_pair}
                ]
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, n=cfg.seed_number-len(prompts), temperature=0.7)
            answers = []
            for answer in response["choices"]:
                answers.append(answer['message']['content'].strip().split('```')[1].split('```')[0])
            success=1
        except Exception  as e:
            info = e.args[0]
            fail_count+=1
            if 'Max retries exceeded with url:' in info:
                sleep(2*fail_count)
            print(info)
        if fail_count>10:
            print('{} fail more than 10 times'.format(str(fail_count)))
            break
    return prompts+answers



def main(cfg, prompts):
    prompts = generate_seed_prompt(cfg, prompts)

    for ite in range(cfg.iteration_number):
        bug_ids = build_d4j_train()
        results = []
        lines = []
        conditions = []
        for idx in range(len(prompts)):
            ut_generation(cfg, bug_ids, prompts[idx])
            print('prompt: {}'.format(prompts[idx]))
            line_coverage, condition_coverage = evaluate_coverage(bug_ids, prompts[idx])
            print('average line coverage: {}, average condition coverage: {}'.format(line_coverage, condition_coverage))
            lines.append(line_coverage)
            conditions.append(condition_coverage)
            results.append(line_coverage+condition_coverage)
        print('best line coverage: {}, best condition coverage: {}'.format(max(lines), max(conditions)))
        indices = sorted(range(len(results)), key=lambda i: results[i], reverse=True)[:3]
        selected_prompts = [prompts[i] for i in indices]
        prompts = prompt_improvement(cfg, selected_prompts)


    bug_ids = build_d4j_test()
    lines = []
    conditions = []
    for idx in range(len(prompts)):
        ut_generation(cfg, bug_ids, prompts[idx])
        print('prompt: {}'.format(prompts[idx]))
        line_coverage, condition_coverage = evaluate_coverage(bug_ids, prompts[idx])
        print('average line coverage: {}, average condition coverage: {}'.format(line_coverage, condition_coverage))
        lines.append(line_coverage)
        conditions.append(condition_coverage)
    print('best line coverage: {}, best condition coverage: {}'.format(max(lines), max(conditions)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_test_cases", default=10, type=int, required=True)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--seed_number", default=5, type=int, required=True)
    parser.add_argument("--generated_number", default=2, type=int, required=True)
    parser.add_argument("--model_api", default=None, type=str, required=True)
    parser.add_argument("--iteration_number", default=5, type=int, required=True)
    parser.add_argument("--seed_prompt_addr", default='seed_prompt.txt', type=str, required=True)

    cfg = parser.parse_args()
    llm_api_key = cfg.model_api
    with open(cfg.seed_prompt_addr) as f:
        prompts = f.readlines()
    seed_prompt = []
    for prompt in prompts:
        if len(prompt.strip())>0:
            seed_prompt.append(prompt)
    main(cfg, seed_prompt)



