#!/bin/bash
set -e;

to_words=0;
if [ "$1" == "--to-words" ]; then to_words=1; shift; fi;

gawk '{
  for (i=2;  i <= NF && ($i == "<space>" || $i == "<dummy>"); ++i) $i="";
  for (i=NF; i > 1   && ($i == "<space>" || $i == "<dummy>"); --i) $i="";
  print;
}' $@ |
gawk -v to_words="$to_words" '{
  if (to_words == 1) {
    printf("%s ", $1);
    for (i=2;i<=NF;++i) {
      if ($i == "<space>")
        printf(" ");
    else
        printf("%s", $i);
    }
    printf("\n");
  } else {
    print;
  }
}'
