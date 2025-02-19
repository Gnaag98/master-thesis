#!/bin/bash

run () {
    local dim_x=$1
    local dim_y=$1
    local dim_z=1
    local cell_size=64
    local particles_per_cell=16
    local version=$2
    local output="../output"
    local distribution=$3
    local should_save=0
    ./master_thesis ${dim_x} ${dim_y} ${dim_z} ${cell_size} \
        ${particles_per_cell} ${version} ${output} ${distribution} \
        ${should_save}
}

get_duration () {
    local dim=$1
    local version=$2
    local distribution=$3
    if [[ ${version} -eq 0 ]]; then
        local time_keyword="Global took"
    else
        local time_keyword="Shared took"
    fi
    run ${dim} ${version} ${distribution} | grep -i "${time_keyword}" | tr -cd 0-9
}

echo_with_comma () {
    local IFS=","
    echo "$*"
}

distribution=${1:-0}
dimensions=(16 32 64 128 256 512 1024 2048)

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

duration_global=()
duration_shared=()
for dim in "${dimensions[@]}"; do
    echo "${dim}x${dim}"
    echo -n "  global "
    version=0
    duration_global+=($(get_duration ${dim} ${version} ${distribution}))
    echo "done"
    echo -n "  shared "
    version=1
    duration_shared+=($(get_duration ${dim} ${version} ${distribution}))
    echo "done"
done

cd ..
filepath=output/durations_both.csv
> ${filepath}
echo_with_comma ${dimensions[@]} >> ${filepath}
echo_with_comma ${duration_global[@]} >> ${filepath}
echo_with_comma ${duration_shared[@]} >> ${filepath}

exit 0
