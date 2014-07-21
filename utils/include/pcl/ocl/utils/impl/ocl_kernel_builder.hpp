#ifndef __OCL_KERNEL_BUILDER__
#define __OCL_KERNEL_BUILDER__

#include <iostream>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <iterator>
#include <numeric> 

using namespace std;

class Parser
{
  public:
    Parser(std::string& type) : _type(type)
    {
	this->point = "PointT";
        //std::cout << "Parsing OpenCL kernel for " << type << std::endl;
    }
    inline void setFile(std::string& filename, std::string& _struct)
    {
		std::ifstream programFile((char*) filename.c_str());
	 	std::string programString(std::istreambuf_iterator<char>(programFile),
          (std::istreambuf_iterator<char>()));
        program = _struct + ReplaceString(programString);
    }
    
    inline std::string ReplaceString(std::string subject) {
		size_t pos = 0;
		while ((pos = subject.find(point, pos)) != std::string::npos) {
			subject.replace(pos, point.length(), _type);
			pos += _type.length();
		}
		return subject;
	}
    
    inline std::string getProgram()
    {
        return program;
    }
  private:
    std::string _type, program;
    std::string point;// = "PointT";

};

class OCLBuilder
{
  public:
    virtual void configureFile(std::string&) = 0;
    Parser *getResult()
    {
        return _result;
    }
  protected:
    Parser *_result;
};

class XYZOCLBuilder: public OCLBuilder
{
  public:
    XYZOCLBuilder()
    {
        type = "PointXYZ";
        _struct = "typedef struct {\n union {\n float data[4];\n struct {\n float x; \n float y; \n float z; \n float w; \n }; \n }; \n } PointXYZ; \n\n";
        _result = new Parser(type);
    }
    inline void configureFile(std::string& filename)
    {
        _result->setFile(filename, _struct);
    }
  private:
	std::string type;
	std::string _struct;
};

class XYZRGBAOCLBuilder: public OCLBuilder
{
  public:
    XYZRGBAOCLBuilder()
    {
	type = "PointXYZRGBA";
        _struct = "typedef struct {\n union {\n float data[4];\n float rgba[4];\n struct {\n float x;\n float y;\n float z;\n float w;\n};\n struct {\n float r;\n float g;\n float b;\n float a;\n};\n};\n} PointXYZRGBA;\n\n";
        _result = new Parser(type);
    }
    inline void configureFile(std::string& filename)
    {
        _result->setFile(filename, _struct);
    }
  private:
    	std::string type;
    	std::string _struct;
};

class Reader
{
  public:
    inline void setOCLBuilder(OCLBuilder *b)
    {
        _OCLBuilder = b;
    }
   inline  void construct(std::string& filename);
  private:
    OCLBuilder *_OCLBuilder;
};

inline void Reader::construct(std::string& filename)
{
      _OCLBuilder->configureFile(filename);
}

#endif
