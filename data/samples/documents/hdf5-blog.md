You probably didn't expect to be reading about data formats today, eh? Turns out this format has an interesting history, and some pretty handy use-cases for large-scale Machine Learning (ML) applications too.

As you may know, manipulating large arrays of numeric values is at the heart of the work of many modern scientific- and data-oriented software professionals. By extension, persisting these arrays is a fundamentally important challenge. Whether its trillions of individual data points from a globe-spanning satellite network, through to the billions of parameters in your latest and greatest neural network. Efficient ways of reading, writing and managing this type of data are clearly going to be a valuable asset.

This post introduces the basics of HDF5 (a data format, data model and set of tools known collectively as HDF5) designed to address precisely these problems, and – to mix it up a bit – outlines how to get started with HDF5 in Go (for reasons that will become clear in future posts). It turns out understanding HDF5 can come in very handy when working on large, modern ML problems...

What do NASA, Lucasfilm and TensorFlow have in common?
Time for a (brief) history lesson. In 1986, as part of a national initiative in the US to boost the availability and quality of supercomputers (and supercomputer infrastructure) the US government founded the National Center for Supercomputing Applications (NCSA) as one of a handful of Supercomputing Centers across the US.

The NCSA would go on to play an important role in the development of the modern web browser. If you're into your tech history, Marc Andreessen worked at NCSA on an early browser known as Mosaic. Andreessen would go on to found Netscape (and the successful Andreessen and Horowitz Venture Capital firm to boot), and elements of Mosaic would be licensed to early versions of Internet Explorer.


A screenshot of the Mosaic web browser, from way back when browsers didn't need 4GB to function properly. From ZDNet.
While the development of early browser technology is arguably one of the projects the NCSA is best known for, there was also a highly influential project in the late 1980's and early 1990's focussed on the development of portable data formats for large-scale scientific problems. The aim was to develop a standard to allow huge volumes of data to be stored, organised and accessed across multiple machines. This was the genesis of the Hierarchical Data Format (HDF) data format.

The aim was to develop a standard to allow huge volumes of data to be stored, organised and accessed across multiple machines. This was the genesis of the Hierarchical Data Format (HDF) data format.
One of the early use-cases for the new HDF format ended up being in one of NASA's flagship projects through the 1990's and early 2000's: the Earth Observing System (EOS). This project involved delivering several specialised satellites into orbit to – you guessed it – observe specific aspects of Earth's atmosphere, land, oceans and biosphere. The project has gathered vast quantities of numerical data, and (unsurprisingly) storing, organising and accessing this data is a critical aspect of the entire project.

During this period, the HDF format was also adopted in many scientific computing applications, including by many national laboratories and other research institutions. It was also adopted by many business too, from banks to mathematics software businesses (including MathWork's MATLAB). Lucasfilm, a company known (in part) for extensive use of CGI animation (and an obscure space-opera saga) developed an interchange graphics format built on HDF in collaboration with Sony.

Fast forward to today, and the latest version of this file format (HDF5) is in use across a huge number of scientific and commercial applications. This includes a number of ML software systems and packages, perhaps most prominently as a key aspect of TensorFlow's mechanism for saving and loading neural network architectures and their associated parameters. HDF5's pedigree with massive scientific datasets lends itself well to precisely this type of problem. It is ideal for persisting custom models, simulation results or environments common to modern ML applications.

Digging deeper
So what makes HDF5 so useful? For one thing, its Data Model makes storing heterogenous data extremely easy. In other words: storing oddly shaped data of mixed types is straightforward. This can be difficult in formats that require table-structured data, or for data to have the same type. In this HDF5 Data Model there are two particularly important 'objects': Groups and Datasets.

Groups are bit like directories in a standard filesystem: you can nest them in arbitrary tree-like structures. In a similar way to a filesystem, there's a 'root' group that contains all other Groups and Datasets. In the filesystem analogy, Datasets are akin to the files themselves.


Figure 1: A high-level overview of the relationship between two key aspects of the HDF5 data model: Groups (blue) and Datasets (green). Note that in combination, these two objects act a lot like a Unix-style filesystem.
A Dataset can contain the raw data (perhaps neural network parameters, or sensor readings), as well as a bunch of arbitrary metadata too. For example, a Dataset can have Attributes – key-value pairs associated with. It also has a defined Datatype which, as you may expect, indicates the type of the Dataset. Finally, a Dataset also has Properties and a Dataspace. The former defines things such as how a Dataset can be accessed (potentially improving read times, for example), while the latter defines the layout of the Dataset. If you're inclined to want more details, there's a more in depth exploration on the HDF5 project page.

Taken together, this data model (and the libraries that implement it) enable data stored in HDF5 format to provide better performance on some tasks than common SQL databases, and a more natural format for numerical array-heavy data structures (like ML models!).

Code time
How about a motivating example. Let's assume you have a collection of raw images data you'd like to load and manipulate. Now you might think Go is an odd choice of language for manipulating data of this kind. However, Go is a language designed for the large-scale multicore and distributed computing applications typical of modern cloud environments. It can be a productive pattern to use a language like Go for specific business-critical/high-performance parts of infrastructure, and a language like Python or R for less performance-critical aspects of the architecture.


Figure 2: A simplified view of a Dataset and associated metadata captured as a collections Attributes. Note that these can provide additional data, for instance shape, format, geotags etc. related to the given Dataset.
In this example, you'll be loading a very simple Dataset from a file. In this case, it's a single image in the form of a raw 1D array of floating point numbers. This Dataset has three Attributes: height, width, and timestamp. The first two Attributes are straightforward enough: the height and width of the image stored in the 1D array. This can allow you to reconstruct the 'original' 2D image array downstream, for example. There's also a timestamp. This is simply the time the image data was created.

There's a Go wrapper for the C implementation of the HDF5 libraries that allow you to read and write HDF5 files. This is part of the Gonum numerical packages for Go ecosystem. This is helpful because it allows you to load data generated and stored in HDF5 format from another system into your Go code very easily. For example, as you read earlier, Python libraries like Tensorflow store Deep Learning models in HDF5 format. Can you see where this is going?

Anyway, as the Gonum HDF5 library has relatively sparse documentation, this post will focus on showing you how to read and write HDF5 data in pure Go. Stay tuned for another post on how to get Go and Python to play nicely together in data- and ML-applications.

Time for some examples...

Writing data
First up: writing some sample data to file! You'll need to generate your 'image data'. For convenience, this'll be a one-dimensional array in this instance (you can always reshape it later). Here's a function (with imports for later functions) for generating a one-dimensional array:

package main

import (
	"fmt"
	"gonum.org/v1/hdf5"
	"math/rand"
	"time"
)


// generate an array of random numbers in range
func randomFloat64(low float64, high float64, size int) []float64 {
	res := make([]float64, size)
	for i := range res {
		res[i] = low + rand.Float64() * (high - low)
	}
	return res
}
Now for the main event. Here's a function that writes an array and associated attributes to a given HDF5 file:

// write image data to HDF5 file
func writeImageData(fileName string, image []float64, height int, width int) {
	// create a new HDF5 file
	file, err := hdf5.CreateFile(fileName, hdf5.F_ACC_TRUNC)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// create a Dataspace of size 'len(image)'
	dims := []uint{uint(len(image))}
	space, err := hdf5.CreateSimpleDataspace(dims, nil)
	if err != nil {
		panic("Failed to create image Dataspace")
	}

	// create a Datatype from the image type
	dtype, err := hdf5.NewDatatypeFromValue(image[0])
	if err != nil {
		panic("Failed to create dtype")
	}

	// create the Dataset itself
	dataset, err := file.CreateDataset("data", dtype, space)

	// write the image data to the dataset
	err = dataset.Write(&image)
	if err != nil {
		panic("Failed to write image data")
	}

	scalar, err := hdf5.CreateDataspace(hdf5.S_SCALAR)
	if err != nil {
		panic("Failed to create scalar dataspace")
	}

	var timestamp = time.Now().UnixNano() / 1000000000  // time in seconds

	heightAttr, err := dataset.CreateAttribute("height", hdf5.T_NATIVE_UINT32, scalar)
	heightAttr.Write(&height, hdf5.T_NATIVE_UINT32)
	heightAttr.Close()

	widthAttr, err := dataset.CreateAttribute("width", hdf5.T_NATIVE_UINT32, scalar)
	widthAttr.Write(&width, hdf5.T_NATIVE_UINT32)
	widthAttr.Close()

	timeAttr, err := dataset.CreateAttribute("timestamp", hdf5.T_NATIVE_FLOAT, scalar)
	timeAttr.Write(&timestamp, hdf5.T_NATIVE_FLOAT)
	timeAttr.Close()

	// close dataset & file
	dataset.Close()
	file.Close()

}
As you can see, this creates a new file, initialises Dataspaces for both the 'main' data itself, and the attributes. It then writes the image data, followed by the attribute data (height, width, timestamp) to file. Lastly, it cleans up by closing all files.

Finally, you can run this little example in your main function:

func main(){
	height := 10
    width := 10
	arr := randomFloat64(0.0, 1.0, height * width)
	writeImageData("image.h5", arr, height, width)
	fmt.Println(arr)
}
And that's it for this minimal data-writing example. You have your sample image data. Here's a Gist with the full example:

A simple Go module for writing a single array of 64 bit floating point numbers to a HDF5 formatted file (with attributes).
A simple Go module for writing a single array of 64 bit floating point numbers to a HDF5 formatted file (with attributes). - write_hdf5_array.go

Gist
262588213843476

Reading data
Next, time to read the data. In a new file, you'll need to import hdf5 again:

package main

import (
	"gonum.org/v1/hdf5"
)
With that done, you can open your image Dataset with:

// load the image data from the indicated HDF5 file.
func getImageDataset(fileName string) *hdf5.Dataset {
	file, err := hdf5.OpenFile(fileName, hdf5.F_ACC_RDONLY)
	if err != nil {
		panic(err)
	}

	dataset, err := file.OpenDataset("data")
	if err != nil {
		panic(err)
	}

	return dataset
}
You'll also need to unpack each of your attributes too. Here's a convenience function to do that for you:

// load a scalar-valued variable of type int from an attribute of the given name on the given dataset.
func getIntScalarAttr(dataset *hdf5.Dataset, attrName string) int {
	var value int

	attr, err := dataset.OpenAttribute(attrName)
	if err != nil {
		panic(err)
	}

	attr.Read(&value, hdf5.T_NATIVE_UINT32)
	return value
}
Note that you aren't going to be reading the timestamp attribute in this example, but the same process applies. Finally, load your data into your Go program. You can do that with:

func main(){
	// load dataset & dataset attributes
	dataset := getImageDataset("image.h5")
	height := getIntScalarAttr(dataset, "height")
	width := getIntScalarAttr(dataset, "width")

	// create image data slice using height/width attributes
	image := make([]float64, height*width)
	dataset.Read(&image)

	// reshape and process here
}
Here's a Gist with the full example:

A simple Go module for reading a single array of 64 bit floating point numbers from a HDF5 formatted file.
A simple Go module for reading a single array of 64 bit floating point numbers from a HDF5 formatted file. - read_hdf5_array.go

Gist
262588213843476

Finishing up
That's your whistlestop tour of HDF5. As always, there's a lot more to dig into. However, you may be able to see how this format is useful for creating application-specific structures for large datasets, and perhaps even how this could be useful to your own work.

The next post will look at how this functionality can enable some interesting interactions between data/ML applications written in Python, and high performance infrastructure written in Go. Stay tuned.

Further reading
Want to read more on HDF5? Great, here's some useful links: