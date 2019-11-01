#!/bin/bash

board_name=$1
support_board=(DP-8020 DP-N1 ZCU102 ZCU104 Ultra96)
dpu_versions=(1.3.7 1.3.7 1.3.0 1.3.0 1.3.7)

install_fail () {
    echo "Installation failed!" 1>&2
    exit 1
}
print_help() {
    echo "Usage: install.sh board_name"
    echo "The board_name is:"
    for (( i=0; i<$(echo ${#support_board[@]}); i++ )); do
        echo "    ${support_board[$i]}"
    done
}

if [ "$board_name" = "" ]; then
    print_help
    install_fail
fi

verfiy=""
for (( i=0; i<$(echo ${#support_board[@]}); i++ )); do
    if [ "$board_name" = "${support_board[$i]}" ]; then
        verify=true;
    fi
done
if [ "$verify" != "true" ]; then
    echo "Do not support board_name $board_name"
    print_help
    install_fail
fi

echo "Inspect system enviroment..."

echo "[system version]"
sysver=`lsb_release -a | grep Description:`
echo $sysver
sysver=`echo $sysver | awk '{print $3}' | awk -F'.' '{print $1"."$2}' `
echo $sysver


cuda_verfile="/usr/local/cuda/version.txt"
cudnn_verfile="/usr/include/cudnn.h"
cuda_ver=
cudnn_ver=

if [ -f $cuda_verfile ] ; then
   cuda_ver=`cat $cuda_verfile | awk '{ print $3 } ' `
else
    echo $cuda_verfile not exist !
    echo  "Begin to install DeePhi DNNDK tools(CPU version) on host ..."
    cd pkgs/ubuntu${sysver} || install_fail
    cp * /usr/local/bin
    chmod og+rx /usr/local/bin/decent || install_fail 
    chmod og+rx /usr/local/bin/dnnc* || install_fail
    [ -f /usr/local/bin/dnnc ] && rm /usr/local/bin/dnnc
    [ -L /usr/local/bin/dnnc ] && rm /usr/local/bin/dnnc
    for (( i=0; i<${#support_board[@]}; i++ )); do
    	if [ "$board_name" = "${support_board[$i]}" ]; then
        	ln -s "dnnc-dpu${dpu_versions[$i]}" /usr/local/bin/dnnc || install_fail
        fi
    done
    echo "Complete installation successfully."
    exit 0
fi

if [ -f $cudnn_verfile ] ; then
cudnn_ver=`cat $cudnn_verfile | grep CUDNN_MAJOR -A 2 | head -3 | awk '{ver=ver$3"."}END{print ver}'`
else
    echo $cudnn_verfile not exist!
    echo  "Begin to install DeePhi DNNDK tools(CPU version) on host ..."
    pwd
    cd pkgs/${sysver} || install_fail
    cp * /usr/local/bin 
    chmod og+rx /usr/local/bin/decent || install_fail
    chmod og+rx /usr/local/bin/dnnc* || install_fail
    [ -f /usr/local/bin/dnnc ] && rm /usr/local/bin/dnnc
    [ -L /usr/local/bin/dnnc ] && rm /usr/local/bin/dnnc
    for (( i=0; i<${#support_board[@]}; i++ )); do
   	 if [ "$board_name" = "${support_board[$i]}" ]; then
                ln -s "dnnc-dpu${dpu_versions[$i]}" /usr/local/bin/dnnc || install_fail
         fi
    done
    echo "Complete installation successfully."
    exit 0
fi


echo "[CUDA version]"
echo $cuda_ver
cuda_ver=`echo $cuda_ver | awk -F'.' '{print $1"."$2}'`
echo "[CUDNN version]"
cudnn_ver=`echo $cudnn_ver | awk -F'.' '{print $1"."$2"."$3}'`
echo $cudnn_ver
########################################################
array=(
    ubuntu14.04/cuda_8.0.61_GA2_cudnn_v7.0.5
    ubuntu16.04/cuda_8.0.61_GA2_cudnn_v7.0.5
    ubuntu16.04/cuda_9.0_cudnn_v7.0.5
    ubuntu16.04/cuda_9.1_cudnn_v7.0.5
 )
for data in ${array[@]}
do
    echo ">>>>>>>>>>>>>>>>>>>>"
    echo $data
    echo $sysver
    echo $cuda_ver
    echo $cudnn_ver
    echo "<<<<<<<<<<<<<<<<<<<<<<"
    if [[ $data =~ $sysver ]] && [[ $data =~ $cuda_ver ]] && [[ $data =~ $cudnn_ver ]] ; then
        echo  "Begin to install DeePhi DNNDK tools on host ..."
        cd pkgs/${data} || install_fail
        cp * /usr/local/bin || install_fail
        for item in *; do
            chmod og+rx "/usr/local/bin/$item" || install_fail
        done
        [ -f /usr/local/bin/dnnc ] && rm /usr/local/bin/dnnc
        [ -L /usr/local/bin/dnnc ] && rm /usr/local/bin/dnnc
        for (( i=0; i<${#support_board[@]}; i++ )); do
            if [ "$board_name" = "${support_board[$i]}" ]; then
                ln -s "dnnc-dpu${dpu_versions[$i]}" /usr/local/bin/dnnc || install_fail
            fi
        done
        echo "Complete installation successfully."
        exit 0
    fi
done

echo "Fail to install DNNDK tools(GPU version) on host machine."
echo "The host system environment supported is as below: "
echo "1 - Ubuntu 14.04 + CUDA 8.0 + cuDNN 7.05"
echo "2 - Ubuntu 16.04 + CUDA 8.0 + cuDNN 7.05"
echo "3 - Ubuntu 16.04 + CUDA 9.0 + cuDNN 7.05"
echo "4 - Ubuntu 16.04 + CUDA 9.1 + cuDNN 7.05"
install_fail

