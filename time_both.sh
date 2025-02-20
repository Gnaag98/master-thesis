#!/bin/bash

run () {
    local dim_x=$1
    local dim_y=$1
    local dim_z=1
    local cell_size=64
    local particles_per_cell=$2
    local version=$3
    local output="../output"
    local distribution=$4
    local should_save=0
    ./master_thesis ${dim_x} ${dim_y} ${dim_z} ${cell_size} \
        ${particles_per_cell} ${version} ${output} ${distribution} \
        ${should_save}
}

get_duration () {
    local dim=$1
    local particles_per_cell=$2
    local version=$3
    local distribution=$4
    if [[ ${version} -eq 0 ]]; then
        local time_keyword="Global took"
    else
        local time_keyword="Shared took"
    fi
    run ${dim} ${particles_per_cell} ${version} ${distribution} | grep -i "${time_keyword}" | tr -cd 0-9
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
duration_global=()
duration_shared=()
for ((dim = ${min_dimensions}; dim <= ${max_dimension}; dim += 128)) do
    dimensions+=(${dim})
    echo "${dim}x${dim}"
    echo -n "  global "
    version=0
    duration_global+=($(get_duration ${dim} ${particles_per_cell} ${version} ${distribution}))
    echo "done"
    echo -n "  shared "
    version=1
    duration_shared+=($(get_duration ${dim} ${particles_per_cell} ${version} ${distribution}))
    echo "done"
done

cd ..
filepath=output/durations_both_${particles_per_cell}ppc.csv
> ${filepath}
echo_with_comma ${dimensions[@]} >> ${filepath}
echo_with_comma ${duration_global[@]} >> ${filepath}
echo_with_comma ${duration_shared[@]} >> ${filepath}

exit 0
