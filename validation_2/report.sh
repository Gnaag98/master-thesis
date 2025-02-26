#!/bin/bash

parse_row () {
    local row=$1
    # Remove everything matching '* ', i.e. only keep anything after the space.
    local value=${row#* }
    echo "${value}"
}

parse_output () {
    # Echo header if there is nothing to parse.
    if [[ $# -eq 0 ]]; then
        printf '"particles","expected","computed"\n'
        return 0
    fi
    local output=$( echo "$1" | grep -P '^\S+:' )
    local particles=$( parse_row "$( echo "${output}" | grep "count" )" )
    local expected=$( parse_row "$( echo "${output}" | grep "expected" )" )
    local computed=$( parse_row "$( echo "${output}" | grep "computed" )" )
    printf '%s,%s,%s\n' "${particles}" "${expected}" "${computed}"
}

# Parse command line arguments.
usage="Usage: ${BASH_SOURCE[0]} version iterations"
if [[ $# -lt 1 ]]; then
    echo ${usage}
    exit 1
fi
version=$1
max_iterations=$2

distribution=uniform

# Get directory of script independent of working directory.
directory=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root=${directory}/..

# Exit early if the program doesn't compile.
${root}/build.sh
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Get current date and time
printf -v date '%(%Y-%m-%d)T' -1
printf -v time '%(%H.%M)T' -1

output_directory=${directory}/output/${date}/${time}
mkdir -p ${output_directory}

create_file () {
    local particles_per_cell=$1
    filename=${version}_${distribution}_${particles_per_cell}ppc.csv
    filepath=${output_directory}/${filename}
    # Create file with headers.
    printf  '"dim_x","dim_y","i",%s\n' "$( parse_output )" > ${filepath}
    echo "${filepath}"
}

loop () {
    local dim_x=$1
    local dim_y=$2
    local particles_per_cell=$3
    local filepath=$4
    local dim_z=1
    for ((i = 0; i < max_iterations; i++)); do
        # Make sure each iteration uses a different seed.
        local random_seed=$RANDOM
        printf '%dx%d ppc=%d i=%d\n' ${dim_x} ${dim_y} ${particles_per_cell} ${i}
        local output=$( \
            ${directory}/run.sh ${dim_x} ${dim_y} ${dim_z} -d ${distribution} \
            -p ${particles_per_cell} -v ${version} -r ${random_seed}
        )
        if [[ $? -ne 0 ]]; then
            exit 1
        fi
        local result=$( parse_output "${output}" )
        echo "${dim_x},${dim_y},${i},${result}" >> ${filepath}
    done
}

particles_per_cell=11
filepath=$( create_file ${particles_per_cell} )
for dim in 256 512 1024 2048; do
    loop ${dim} ${dim} ${particles_per_cell} ${filepath}
done

particles_per_cell=13
filepath=$( create_file ${particles_per_cell} )
for dim in 256 512 1024 2048; do
    loop ${dim} ${dim} ${particles_per_cell} ${filepath}
done

particles_per_cell=16
filepath=$( create_file ${particles_per_cell} )
for dim in 256 512 1024 2048; do
    loop ${dim} ${dim} ${particles_per_cell} ${filepath}
done

particles_per_cell=17
filepath=$( create_file ${particles_per_cell} )
for dim in 256 512 1024 2048; do
    loop ${dim} ${dim} ${particles_per_cell} ${filepath}
done

particles_per_cell=19
filepath=$( create_file ${particles_per_cell} )
for dim in 256 512 1024 2048; do
    loop ${dim} ${dim} ${particles_per_cell} ${filepath}
done

particles_per_cell=64
filepath=$( create_file ${particles_per_cell} )
for dim in 128 256 512 1024; do
    loop ${dim} ${dim} ${particles_per_cell} ${filepath}
done

particles_per_cell=17
filepath=$( create_file "prime_${particles_per_cell}" )

dim_x=1048573
dim_y=1
loop ${dim_x} ${dim_y} ${particles_per_cell} ${filepath}

dim_x=1563
dim_y=1861
loop ${dim_x} ${dim_y} ${particles_per_cell} ${filepath}

dim_x=1
dim_y=1048573
loop ${dim_x} ${dim_y} ${particles_per_cell} ${filepath}
