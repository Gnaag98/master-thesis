#!/bin/bash

run () {
    local dim_x=$1
    local dim_y=$1
    local dim_z=1
    local cell_size=64
    local particles_per_cell=$2
    local version=1
    local output="../output"
    local distribution=$3
    local should_save=0
    ./master_thesis ${dim_x} ${dim_y} ${dim_z} ${cell_size} \
        ${particles_per_cell} ${version} ${output} ${distribution} \
        ${should_save}
}

get_duration () {
    local program_output=$1
    local time_keyword="$2 took"
    echo "${program_output}" | grep -i "${time_keyword}" | tr -cd 0-9
}

echo_with_comma () {
    local IFS=","
    echo "$*"
}

distribution=${1:-0}
particles_per_cell=${2:-16}

./build.sh
if [ $? -ne 0 ]; then
    exit 1
fi

cd build

echo -n "Selected particle distribution: "
case ${distribution} in
    0)
        echo uniform
        ;;
    1)
        echo 2D pattern
        ;;
esac

echo "Selected particles per cell: ${particles_per_cell}"

if [[ ${particles_per_cell} -le 16 ]]; then
    min_dimensions=1024
    max_dimension=2048
elif [[ ${particles_per_cell} -le 32 ]]; then
    min_dimensions=640
    max_dimension=1536
else
    min_dimensions=512
    max_dimension=1024
fi

dimensions=()
durations_associate=()
durations_sort=()
durations_contextualize=()
durations_density=()
for ((dim = ${min_dimensions}; dim <= ${max_dimension}; dim += 128)) do
    echo -n "${dim}x${dim} "
    program_output=$(run ${dim} ${particles_per_cell} ${distribution})
    echo "done"

    dimensions+=(${dim})
    durations_associate+=($(get_duration "${program_output}" "associate"))
    durations_sort+=($(get_duration "${program_output}" "sort"))
    durations_contextualize+=($(get_duration "${program_output}" "contextualize"))
    durations_density+=($(get_duration "${program_output}" "density"))
done

cd ..
filepath=output/durations_shared_${particles_per_cell}ppc.csv
> ${filepath}
echo_with_comma ${dimensions[@]} >> ${filepath}
echo_with_comma ${durations_associate[@]} >> ${filepath}
echo_with_comma ${durations_sort[@]} >> ${filepath}
echo_with_comma ${durations_contextualize[@]} >> ${filepath}
echo_with_comma ${durations_density[@]} >> ${filepath}

exit 0
