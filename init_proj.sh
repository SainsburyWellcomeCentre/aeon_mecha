#!/bin/bash

# <h
#
# Initializes a project directory according to the specified programming 
# language and options.
# 
# Usage: `init_proj <name> <language> <opt_flag> <opt_flag_arg>`
#
# Input args:
#     name     : The project name (top-level directory).
#     language : The main programming language of the project. Determines which
#                files to create.
#
# Optional input args (flags): 
#     -n | --no-clobber           : Ensures no overwriting of files that 
#                                   already exist. (Default: false)
#     -e | --exclude <file_array> : Exclude creation of the space-separated 
#                                   filenames in the following <file_array> arg.
#                                   (Default: none)
#     -i | --include <file_array> : Include creation of the space-separated 
#                                   filenames in the following <file_array> arg.
#                                   (Default: none)
#     -p | --path                  : The full path to the project's parent dir.
#                                   (Default: current dir)
#
# /h>

# <s Read and check input args

SUPPORTED_LANGUAGES=('python' 'c')

# Check required args.
name=$1
[[ ${name} == '' ]] && echo "Error: No input args provided." && exit 1
language=${2,,}
lang_found=0  # flag for whether user-spec lang is a supported lang

# Ensure specified language is supported.
for l in "${SUPPORTED_LANGUAGES[@]}"; do
	if [[ ${language} == ${l} ]]; then
		lang_found=1  # signifies match found
		break
	fi
done
if (( ! $lang_found )); then  # throw warning if match not found
	echo "Warning: '${language}' is not supported."
	echo "Supported languages are: ${SUPPORTED_LANGUAGES[@]}."
	echo "No language-specific files will be created."
fi

# Check optional flags.
no_clobber_flag=0
excluded_files=()
included_files=()
parent_dir=$(pwd)
n_opt_args=$(( $# - 2 ))
shift 2
while (( n_opt_args )); do
	case "$1" in
		-n|--no-clobber) 
			no_clobber_flag=1
			shift 1
			(( n_opt_args-- ))
			;;
		-e|--exclude) 
			excluded_files=("$2")
			shift 2
			n_opt_args=$(( n_opt_args - 2 ))
			;;
		-i|--include)
			included_files=("$2")
			shift 2
			n_opt_args=$(( n_opt_args - 2 ))
			;;
		-p|--path)
			parent_dir=$2
			shift 2
			n_opt_args=$(( n_opt_args - 2 ))
			if [[ ! -d $parent_dir ]]; then
				echo "The user specified path, '${parent_dir}', does not exist"
				exit 1
			fi
			;;
		*)
			echo "The flag ${1} is not supported"
			exit 1
			;;
	esac
done

# /s>

# <s Create files.

# Delete dir if exists and `no-clobber` is set to 0.
[[ -d "${parent_dir}/${name}" ]] && [[ ! no_clobber_flag ]] \
	&& rm -r "${parent_dir}/${name}"
# Make and cd into project dir.
mkdir -p "${parent_dir}/${name}"
cd "${parent_dir}/${name}"

# Set default file list (regardless of programming language).
file_list=('readme.md' 'license.md' 'contributing.md' 'changelog.md' \
	'docs/readme.md' 'docs/examples/readme.md' 'docs/examples/data/readme.md' \
	'tests/readme.md' 'tests/data/readme.md' '.gitignore' '.gitattributes' \
	'.github/workflows/actions.yml')
file_dirs=('docs/examples/data' 'tests/data' '.github/workflows')

# Get language-specific files.
case ${language} in
	python)
		# python files to add
		add_files=('pyproject.toml' 'setup.py' 'setup.cfg' 'manifest.in' \
			'requirements.txt' 'env.yml' '.travis.yml' '.flake8' 'tox.ini' \
			'makefile' "${name}/readme.md")
		add_files_dirs=("${name}")
		comment_symbol=$"#"
		;;
	c)
		# c files to add
		add_files=('makefile' "src/${name}.c" "src/${name}.h" \
			'bin/readme.md' 'tools/readme.md' 'lib/readme.md' \
			'log/readme.md' 'include/readme.md')
		add_files_dirs=('src' 'bin' 'tools' 'lib' 'log' 'include')
		comment_symbol=$"//"
		;;
esac

# Make necessary dirs.
file_dirs+=(${add_files_dirs[@]})
mkdir -p ${file_dirs[@]}
touch null  # default empty file
# Add language-specific files.
file_list+=(${add_files[@]})
# Add user specified included files.
file_list+=(${included_files[@]})
# Remove user specified excluded files.
for f in "${excluded_files[@]}"; do
	file_list=(${file_list[@]//*${f}*})
done
# Ensure array is unique (`sort` requires elements to be on new lines, so we
# have to go back and forth between delimiter being `' '` and `'\n'`).
file_list=($(printf "%s\n" "${file_list[@]}" | sort -u | tr '\n' ' '))
# Create the files.
for f in "${file_list[@]}"; do
	[[ ! -f ${f} ]] && cp null ${f}
	# Write a comment in each new file to ensure it's not empty.
	printf "%s" "${comment_symbol}" >> ${f}
done
rm null

# /s>
