/**
 * @file FSInput.h
 * 
 * @ingroup Simulator/Utils/Inputs
 *
 * @brief A factoy class for stimulus input classes.
 * 
 * The FSInput creates an instance of stimulus input class object.
 *
 * In the CreateInstance method, it reads stimulus input file and gets input method
 * (under InputParams/IMethod). If the name of the input method is "SInputRegular",
 * it creates the instance of GpuSInputRegular or HostSInputRegular class depending
 * on the type of the device. 
 * If the name of the input method is "SInputPoisson",
 * it creates the instance of GpuSInputPoisson or HostSInputPoisson class depending
 * on the type of the device. 
 */

#pragma once

#ifndef _FSINPUT_H_
   #define _FSINPUT_H_

   #include "ISInput.h"

class FSInput {
public:
   //! The constructor for FSInput.
   FSInput();
   ~FSInput();

   static FSInput *get()
   {
      static FSInput instance;
      return &instance;
   }

   //! Create an instance.
   ISInput *CreateInstance();

protected:
};

#endif   // _FSINPUT_H_
