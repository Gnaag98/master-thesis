#!/bin/bash

run () {
    local dim_x=$1
    local dim_y=$2
    local dim_z=1
    local cell_size=64
    local particles_per_cell=16
    local version=$3
    local output="../output"
    local distribution=$4
    local should_save=0
    ./master_thesis ${dim_x} ${dim_y} ${dim_z} ${cell_size} \
        ${particles_per_cell} ${version} ${output} ${distribution} \
        ${should_save}
}

get_duration () {
    local dim_x=$1
    local dim_y=$2
    local version=$3
    local distribution=$4
    if [[ ${version} -eq 0 ]]; then
        local time_keyword="Global took"
    else
        local time_keyword="Shared took"
    fi
    run ${dim_x} ${dim_y} ${version} ${distribution} | grep -i ${time_keyword} | tr -cd 0-9
}

echo_with_comma () {
    local IFS=","
    echo "$*"
}

dim_x=$1
dim_y=$2
version=$3
distribution=$4
run_count=$5

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

durations=()
echo "Dimensions: ${dim_x}x${dim_y}"
for i in $(seq 1 ${run_count}); do
    echo -n "Iteration ${i} "
    durations+=($(get_duration ${dim_x} ${dim_y} ${version} ${distribution}))
    echo "done"
done

cd ..
filepath=output/durations_repeated.csv
> ${filepath}
echo_with_comma ${durations[@]} >> ${filepath}

exit 0
