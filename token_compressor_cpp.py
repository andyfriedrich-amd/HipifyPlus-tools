import re
import pyparsing as pp

def move_extra_newline(code):
    # Remove excess new lines
    lines = code.split('\n')
    code = '\n'.join(line for line in code.split('\n') if line.strip())

    code = re.sub(r'\n+', '\n', code)

    return code

def remove_comments(code):
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)

    # Remove triple-quoted comments
    code = re.sub(r'r""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    

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
    # code = re.sub(r',\s\s+', ', ', code)

    return code





def shift_code_left(code):
    # Find the number of spaces in the first line
    first_line = code.split('\n')[0]
    num_spaces = len(first_line) - len(first_line.lstrip())

    # Shift all lines to the left by the number of spaces
    lines = code.split('\n')
    shifted_lines = [line[num_spaces:] for line in lines]
    shifted_code = '\n'.join(shifted_lines)

    return shifted_code


# use it if PEP8 is not good enough
def extra_compress(code):
    # Change four consecutive spaces to a single space
    code = re.sub(r' {2}', ' ', code)

    return code


#split each function/class at the same layer into different class
def split_code(code):
    # Split code based on indentation
    lines = code.split('\n')

    file_contents = []
    current_indent = 0
    current_file_content = ''
    for line in lines:
        stripped_line = line.strip()
        indentation = len(line) - len(stripped_line)

        if stripped_line.startswith(('def ', 'class ')) and indentation == 0:
            # Create a new file for functions and classes            
            file_contents.append(current_file_content)
            current_file_content = ''
        
        current_file_content += line
        current_file_content += '\n'

    file_contents.append(current_file_content)

    return file_contents




def minimize_python_code(file_path, output_directory):
    with open(file_path, 'r') as file:
        code = file.read()

    # Remove comments
    #code = remove_comments(code)
    code = fix_multilines(code)

    # Remove extra newline
    #code = move_extra_newline(code)

    # Shift code to the left
    #code = shift_code_left(code)


    code = extra_compress(code)


    #file_contents = split_code(code)
    # Write each functions and classes in a new file
    output_file_path = f"{output_directory}/file{i + 1}.py"
    with open(output_file_path, 'w') as output_file:
        output_file.write(content)


# change your input dir and output dir here
input_file_path = './data/input.py'
output_directory = './data/out'
minimize_python_code(input_file_path, output_directory)