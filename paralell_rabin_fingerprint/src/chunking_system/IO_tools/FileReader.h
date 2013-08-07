/**
 * FileReader.h
 *
 *	This c lass is just a wrapper around the istream object for
 *	reading files. This wrapper is convenient since it keeps the
 *	name and path of the file that was opened, something that the
 *	simple istream does not do.
 *
 *  Created on: Jul 13, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef FILEREADER_H_
#define FILEREADER_H_

#include <string>
#include <fstream>

class FileReader {
private:
	std::string absolutePath; // the path to the file
	std::ifstream inputStream; // the stream

public:
	/**
	 * Construct the object and initializes all the internal variables
	 * @param pathToFile
	 */
	FileReader(std::string pathToFile) :
			inputStream(pathToFile.c_str(), std::ofstream::binary), absolutePath(pathToFile) {
	}

	/**
	 * Closes out the stream
	 */
	virtual ~FileReader() {
		this->inputStream.close();
	}
	/**
	 * Retrieves the file stream
	 *
	 * @return the stream
	 */
	std::ifstream& getStream() {
		return this->inputStream;
	}

	/**
	 * Retrieves the absolute path of the file
	 * @return a string that indicates the absolute path
	 */
	std::string getLocation() {
		return this->absolutePath;
	}

};

#endif /* FILEREADER_H_ */
