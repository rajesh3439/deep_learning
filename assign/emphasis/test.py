import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='My Python script')

# Add arguments
parser.add_argument('-f', '--file', help='Input file', required=True)
parser.add_argument('-o', '--output', help='Output file', required=False)

# Parse the arguments
args = parser.parse_args()

# Access the argument values
input_file = args.file
output_file = args.output

# Rest of your script logic
print(f'Input file: {input_file}')
if output_file:
    print(f'Output file: {output_file}')