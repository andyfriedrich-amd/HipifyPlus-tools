import tiktoken
import re
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

        content = content.replace('\n', '')

        #content = content.replace(' ', '')

        content = content.replace('\x1b[0K', '')

        content = content.replace('\x1b[1G', '')

        content = content.replace('->', '\n')

        content = content.replace(' [ok]', '\n')

        content = content.replace('[ok]', '\n')

        content = content.replace('OneDrive-', 'OneDrive -')

        content = content.replace('-Advanced', '- Advanced')


        with open(out_path, 'w', encoding="utf-8") as file:
            file.write(content)


def code_compression_test(out_path, output_directory):
    print("performing code_compression_test")
    pattern = r"C:(.*?)\n"
    #out_path = 'betterlog.txt'
    counter_1k = 0
    counter_for_compress = 0
    counter_above = 0
    counter_file_miss = 0
    with open(out_path, "r", encoding="utf-8") as file:
        text = file.read()
        #   print(text)
        matches = re.findall(pattern, text)

        for match in matches:
            code_path = os.path.normpath(match)
            print(code_path)
            #try:
            with open(code_path, "r") as file:
                file_contents = file.read()
                tokens = enc.encode(file_contents)
                compressed_content = minimize_cpp_code(file_contents)
                
                #SAVE IT FOR NOW
                output_file_path = os.path.join(output_directory, os.path.basename(code_path).split('/')[-1])
                with open(output_file_path, 'w') as output_file:
                    output_file.write(compressed_content)
                

                compressed_tokens = enc.encode(compressed_content)
                num_tokens = len(tokens)
                compressed_num_tokens = len(compressed_tokens)
                if(compressed_num_tokens <= 1000):
                    counter_1k += 1
                elif(compressed_num_tokens > 1000 and compressed_num_tokens <= 3000):
                    counter_for_compress += 1
                else:
                    counter_above += 1
                
                
                print("Number of tokens:", num_tokens)
                print("Number of tokens AFTER COMPRESS:", compressed_num_tokens)
            #except Exception:
                #print(code_path)
                #print("FILE MISS!!!!")
            #    counter_file_miss += 1
            #    pass
            print()
    print("After compression we have")
    print("less than 1k %d", counter_1k)
    print("Around 3k %d", counter_for_compress)
    print("ABOVE 3k %d", counter_above)
    print("File miss %d", counter_file_miss)



def count_token():
    pattern = r"C:(.*?)\n"
    out_path = 'betterlog.txt'
    counter_1k = 0
    counter_for_compress = 0
    counter_above = 0
    counter_file_miss = 0
    with open(out_path, "r", encoding="utf-8") as file:
        text = file.read()
        #print(text)
        matches = re.findall(pattern, text)

        for match in matches:
            code_path = os.path.normpath(match)
            #code_path = os.path.join('/home/mobaxterm/Desktop/ai4amd', norm_path)
            print(code_path)
            #try:
            with open(code_path, "r") as file:
                file_contents = file.read()
                tokens = enc.encode(file_contents)
                num_tokens = len(tokens)
                if(num_tokens <= 1000):
                    counter_1k += 1
                elif(num_tokens > 1000 and num_tokens <= 3000):
                    counter_for_compress += 1
                else:
                    counter_above +=1
                print("Number of tokens:", num_tokens)
            #except Exception:
                #print(code_path)
                #print("FILE MISS!!!!")
            #    counter_file_miss += 1
            #    pass
            print()
    print("less than 1k %d", counter_1k)
    print("Around 3k %d", counter_for_compress)
    print("ABOVE 3k %d", counter_above)
    print("File miss %d", counter_file_miss)




if __name__ == "__main__":
    enc = tiktoken.encoding_for_model("code-davinci-002")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    file_path = "../data_gen/pytorch/torch_hipify.log"
    out_path = 'betterlog.txt'
    output_directory = './out'
    code_compression_test(out_path, output_directory)
    #count_token()
