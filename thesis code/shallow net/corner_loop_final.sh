Directory="/home/kk/Documents/Semester4/code/Corner_contour_net_shallow"
Priority="0"
echo $Directory
mkdir $Directory/argh_log_3288_fix
for i in {0..20}; do
  for out_channels_idx in {0..0}; do
    for kernel_size_idx in {0..0}; do
      for stride_idx in {0..0}; do
        echo "hostname; cd $Directory ; /home/kk/P3.10/bin/python3 cnn_training.py --idx-conv-out-channels-list $out_channels_idx --idx-conv-kernel-sizes $kernel_size_idx --idx-conv-stride-sizes $stride_idx -s \$JOB_ID" | qsub -o $Directory/argh_log_3288_fix -j y -p $Priority -q gp4u,gp3u -N Corner3288fix
      done
    done
  done
done
