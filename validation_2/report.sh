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
        printf '"expected","computed"\n'
        return 0
    fi
    local output=$( echo "$1" | grep -P '^\S+:' )
    local expected=$( parse_row "$( echo "${output}" | grep "expected" )" )
    local computed=$( parse_row "$( echo "${output}" | grep "computed" )" )
    printf '%s,%s\n' "${expected}" "${computed}"
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
    local dim_x=$1
    local dim_y=$2
    local run_index=$3
    local filename="${version}_${distribution}_${dim_x}x${dim_y}_run_${run_index}.csv"
    local filepath=${output_directory}/${filename}
    # Create file with headers.
    printf  '"i",%s\n' "$( parse_output )" > ${filepath}
    echo "${filepath}"
}

loop () {
    local dim_x=$1
    local dim_y=$2
    local particles_per_cell=$3
    local filepath=$4
    local dim_z=1

    local i
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
        echo "${i},${result}" >> ${filepath}
    done
}

nonrandom_loop () {
    local dim_x=$1
    local dim_y=$2
    local particles_per_cell=$3
    local filepath=$4
    local dim_z=1
    local random_seed=1

    local i
    for ((i = 0; i < max_iterations; i++)); do
        # Make sure each iteration uses a different seed.
        printf '%dx%d ppc=%d i=%d\n' ${dim_x} ${dim_y} ${particles_per_cell} ${i}
        local output=$( \
            ${directory}/run.sh ${dim_x} ${dim_y} ${dim_z} -d ${distribution} \
            -p ${particles_per_cell} -v ${version} -r ${random_seed}
        )
        if [[ $? -ne 0 ]]; then
            exit 1
        fi
        local result=$( parse_output "${output}" )
        echo "${i},${result}" >> ${filepath}
    done
}

# Square and rectangle grids with random seeds.
run_index=1
xs=(256 256 512 512 512 1024 1024)
ys=(256 512 256 512 1024 512 1024)
for ((i = 0; i < ${#xs[@]}; i++)); do
    dim_x=${xs[i]}
    dim_y=${ys[i]}
    for ppc in 16 32; do
        filepath=$( create_file ${dim_x} ${dim_y} ${run_index} )
        loop ${dim_x} ${dim_y} ${ppc} ${filepath}
        ((++run_index))
    done
done

# Square grids with fixed seed.
run_index=1
for dim in 256 512 1024; do
    for ppc in 16 32; do
        filepath=$( create_file "nonrandom_${dim}" ${dim} ${run_index} )
        nonrandom_loop ${dim} ${dim} ${ppc} ${filepath}
        ((++run_index))
    done
done
