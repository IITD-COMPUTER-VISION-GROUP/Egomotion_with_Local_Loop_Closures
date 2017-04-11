function makeSampleFile(start_id, batch_size, bootstrap)

fid = fopen('../config.txt','wt');
fprintf(fid, '%d\n%d\n%d', start_id, batch_size, bootstrap);
fclose(fid);

end
