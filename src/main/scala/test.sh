java -cp "/home/ding/Downloads/stanford-ner-2015-12-09/*:/home/ding/Downloads/stanford-ner-2015-12-09/lib/*" edu.stanford.nlp.process.PTBTokenizer $1/0.txt > $1/token.tok
perl -ne 'chomp; print "$_\tO\n"' $1/token.tok > $1/file.tsv
