Directory="/home/kk/Documents/Semester4/code/Run64Variations"
Priority="0"
echo $Directory
mkdir $Directory/argh_log_corner
for out_channels_idx in {0..63}; do
  for kernel_size_idx in {0..0}; do
    for stride_idx in {0..0}; do
      echo "hostname; cd $Directory ; /home/kk/P3.10/bin/python3 cnn_training.py --idx-conv-out-channels-list $out_channels_idx --idx-conv-kernel-sizes $kernel_size_idx --idx-conv-stride-sizes $stride_idx -s \$JOB_ID" | qsub -o $Directory/argh_log_classic -j y -p $Priority -q gp4u,gp3u -N itsCorn
    done
  done
done
