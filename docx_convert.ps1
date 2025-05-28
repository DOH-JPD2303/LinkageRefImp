# Convert github-flavored markdown to docx for DOH-style publication.
pandoc -f gfm -t docx -o blog.docx README.md
