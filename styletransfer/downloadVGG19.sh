FILE_NAME="vgg19_normalized.pkl"

if [ -f $FILE_NAME ]; then
	echo $FILE_NAME" has already been downloaded"
  exit 0
else
	echo "downloading "$FILE_NAME
	# If the download fails we want to remove the file that was created, so as to
	# not create the illusion that the download was successful.
	wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/$FILE_NAME -O $FILE_NAME || rm -f $FILE_NAME
fi

if [ -f $FILE_NAME ]; then
	echo "Successfully downloaded "$FILE_NAME
  exit 0
else
	echo "Failed to download "$FILE_NAME", please try again"
  exit 1
fi
