import tiktoken
import re
import time
import os

##########################################################################################
# include code for cpp compression

def remove_comments(code):
    # Remove single-line comments
    code = re.sub(r'//.*', '', code)

    # Remove triple-quoted comments
    code = re.sub(r'/[*].*?[*]/', '', code, flags=re.DOTALL)
    #code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    

    return code


def extra_compress(code):
    # Change four consecutive spaces to a single space
    code = re.sub(r' {2}', ' ', code)

    return code

def remove_extra_newline(code):
    # Remove excess new lines
    lines = code.split('\n')
    code = '\n'.join(line for line in code.split('\n') if line.strip())

    code = re.sub(r'\n+', '\n', code)

    return code

def fix_multilines(code):
    # Find all occurrences of content between regular parentheses
    matches1 = re.findall(r'\\\n', code, flags=re.DOTALL)


    for match in matches1:
        tide_content = re.sub(r'\s\s+', ' ', match)
        fixed_content = tide_content.replace('\\\n', ' ')
        code = code.replace(match, fixed_content)


    # Find all occurrences of , + \n
    matches2 = re.findall(r',\n', code, flags=re.DOTALL)

    for match in matches2:
        #print(match)
        fixed_content = match.replace(',\n', ', ')
        code = code.replace(match, fixed_content)

    # replace all the , + \n with comma space
    code = re.sub(r',\s\s+', ', ', code)

    return code


def minimize_cpp_code(code):

    code = remove_extra_newline(code)

    code = remove_comments(code)

    code = fix_multilines(code)

    code = extra_compress(code)

    return code

    #output_file_path = os.path.join(output_directory, os.path.basename(file_path).split('/')[-1])
    #with open(output_file_path, 'w') as output_file:
    #    output_file.write(content)


##########################################################################################
# enter file pathfor hipify.log 
def fix_log(file_path, out_path):
    #enc = tiktoken.get_encoding("p50k_base")



    #file_path = "../data_gen/pytorch/torch_hipify.log"
    #out_path = 'betterlog.txt'
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

        #content = content.replace('\n', '')

        #content = content.replace(' ', '')

        #content = content.replace('\x1b[0K', '')

        #content = content.replace('\x1b[1G', '')

        content = content.replace('-- Converted: ', '')

        content = content.replace('-- Excluded from hipify: ', '')

        content = content.replace(' -> ', '\n')

        with open(out_path, 'w', encoding="utf-8") as file:
            file.write(content + "\n")  

def code_compression_test(out_path, output_directory):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("performing code_compression_test")
    pattern = r"(.*?)\n"
    #out_path = 'betterlog.txt'
    counter_1k = 0
    counter_for_compress = 0
    counter_above = 0
    counter_file_miss = 0
    missed_files = []
    with open(out_path, "r", encoding="utf-8") as file:
        text = file.read()
        #   print(text)
        matches = re.findall(pattern, text)
        count = 0


        print(len(matches))
        time.sleep(5)

        for i in range(0, len(matches), 2):
            match1 = matches[i]
            match2 = matches[i+1]
            code_path1 = os.path.normpath(match1)
            code_path2 = os.path.normpath(match2)
            print(code_path1 + "and" + code_path2)
            try:
                with open(code_path1, "r") as file:
                    file_contents1 = file.read()
                    tokens1 = enc.encode(file_contents1)
                    if(len(tokens1) <= 1005):
                        compressed_content1 = file_contents1
                        compressed_num_tokens1 = len(tokens1)
                    else:
                        compressed_content1 = minimize_cpp_code(file_contents1)
                        compressed_tokens1 = enc.encode(compressed_content1)
                        compressed_num_tokens1 = len(compressed_tokens1)
                
                with open(code_path2, "r") as file:
                    file_contents2 = file.read()
                    tokens2 = enc.encode(file_contents2)
                    if(len(tokens2) <= 1005):
                        compressed_content2 = file_contents2
                        compressed_num_tokens2 = len(tokens2)
                    else:
                        compressed_content2 = minimize_cpp_code(file_contents2)
                        compressed_tokens2 = enc.encode(compressed_content2)
                        compressed_num_tokens2 = len(compressed_tokens2)
            except Exception:
                counter_file_miss += 2
                print("Something goes wrong")
                continue
        
                
                #SAVE IT FOR NOW

            if(compressed_num_tokens1 <= 1005 and compressed_num_tokens2 <= 1005):
                counter_1k += 1
                output_file_path1 = os.path.join(output_directory, str(count) + "_" + "c" + os.path.basename(code_path1).split('/')[-1])
                output_file_path2 = os.path.join(output_directory, str(count) + "_" + "h" + os.path.basename(code_path2).split('/')[-1])
                count +=1
                with open(output_file_path1, 'w') as output_file1:
                    output_file1.write(compressed_content1)
                with open(output_file_path2, 'w') as output_file2:
                    output_file2.write(compressed_content2)
                if(abs(compressed_num_tokens1 - compressed_num_tokens2) > 50):
                    print("WHAT THE FUCK??????????")
            
            else:
                counter_file_miss += 1

            print("Number of tokens:%d and %d", compressed_num_tokens1, compressed_num_tokens2)
                #print("Number of tokens AFTER COMPRESS:", compressed_num_tokens)
            #except Exception:
                #print(code_path)
                #print("FILE MISS!!!!")
            #    counter_file_miss += 1
            #    pass
            print()
    print("After compression we have")
    print("less than 1k %d", counter_1k)
    #print("Around 3k %d", counter_for_compress)
    #print("ABOVE 3k %d", counter_above)
    print("File miss %d", counter_file_miss)


if __name__ == "__main__":
    enc = tiktoken.encoding_for_model("code-davinci-002")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    file_path = "../hipify.log"
    out_path = '../betterlog.log'
    output_directory = './out3'
    fix_log('../hipify.log', '../betterlog.log')
    code_compression_test(out_path, output_directory)
    
    # file_path_2 = '../skipped_hipify.log'
    # out_path_2 = '../betterskip.log'
    # fix_log(file_path_2, out_path_2)
    # code_compression_test(out_path_2, output_directory)
    # #count_token()

    