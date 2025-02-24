#!/bin/bash

get_duration () {
    local version=$1
    shift
    if [[ ${version} -eq 0 ]]; then
        local time_keyword="Global took"
    else
        local time_keyword="Shared took"
    fi
    build/master_thesis $* -v ${version} \
        | grep -i "${time_keyword}" \
        | tr -cd 0-9
}

echo_with_comma () {
    local IFS=","
    echo "$*"
}

./build.sh
if [ $? -ne 0 ]; then
    exit 1
fi

distribution=uniform
particles_per_cell=16

# Parse arguments to remove those overwritten by this script.
args=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -d)
            # Replace default with user defined value.
            distribution=$2
            shift
            shift
            ;;
        -p)
            # Replace default with user defined value.
            particles_per_cell=$2
            shift
            shift
            ;;
        -v)
            # No reason to specify a version when both version will be run.
            shift
            shift
            ;;
        *)
            args+=($1)
            shift
            ;;
    esac
done

echo -n "Selected particle distribution: "
case ${distribution} in
    uniform)
        echo uniform
        ;;
    pattern_2d)
        echo 2D pattern
        ;;
    *)
        echo file
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
    version=global
    duration_global+=($(get_duration ${version} ${args[*]} -p ${particles_per_cell} -d ${distribution}))
    echo "done"
    echo -n "  shared "
    version=shared
    duration_shared+=($(get_duration ${version} ${args[*]} -p ${particles_per_cell} -d ${distribution}))
    echo "done"
done

filepath=output/durations_both_${particles_per_cell}ppc.csv
> ${filepath}
echo_with_comma ${dimensions[@]} >> ${filepath}
echo_with_comma ${duration_global[@]} >> ${filepath}
echo_with_comma ${duration_shared[@]} >> ${filepath}

exit 0
