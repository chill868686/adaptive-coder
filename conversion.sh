#!/bin/bash

echo $@
while [ -n "$1" ]
    do
      name=${1%=*}
      name=${name#*--}
      para=${1#*=}
      shift 1
        case "$name" in
            adaptive_coder_path)
	              acp="$para"
                echo $acp
                ;;
            file_path)
                fp="$para"
                echo $fp
                ;;
            coding_type)
                ct="$para"
                echo $ct
                ;;
            log)
                lg="$para"
                echo lg
                ;;
        esac
    done

python $acp/DNACoder.py --adaptive_coder_path=$acp --file_path=$fp --coding_type=$ct --log=$lg