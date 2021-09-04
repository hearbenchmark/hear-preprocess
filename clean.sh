#!/bin/sh
# Clean all directories besides the download directory.
rm -Rf _workdir/*/0[2-9]*
rm -Rf _workdir/*/[1-9]*
rm -Rf tasks/
rm hear-2021*.tar.gz
