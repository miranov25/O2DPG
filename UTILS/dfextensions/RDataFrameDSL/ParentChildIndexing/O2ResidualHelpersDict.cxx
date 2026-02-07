// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME O2ResidualHelpersDict
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "ROOT/RConfig.hxx"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Header files passed as explicit arguments
#include "o2_residual_helpers.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace RDataFrameDSL {
   namespace ROOTDict {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *RDataFrameDSL_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("RDataFrameDSL", 0 /*version*/, "o2_residual_helpers.h", 23,
                     ::ROOT::Internal::DefineBehavior((void*)nullptr,(void*)nullptr),
                     &RDataFrameDSL_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_DICT_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_DICT_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *RDataFrameDSL_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}

namespace RDataFrameDSL {
   namespace O2ResidualHelpers {
   namespace ROOTDict {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *RDataFrameDSLcLcLO2ResidualHelpers_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("RDataFrameDSL::O2ResidualHelpers", 0 /*version*/, "o2_residual_helpers.h", 24,
                     ::ROOT::Internal::DefineBehavior((void*)nullptr,(void*)nullptr),
                     &RDataFrameDSLcLcLO2ResidualHelpers_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_DICT_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_DICT_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *RDataFrameDSLcLcLO2ResidualHelpers_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}
}

namespace ROOT {
   static TClass *ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR_Dictionary();
   static void ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR_TClassManip(TClass*);
   static void *new_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(void *p = nullptr);
   static void *newArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(Long_t size, void *p);
   static void delete_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(void *p);
   static void deleteArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(void *p);
   static void destruct_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>*)
   {
      ROOT::VecOps::RVec<o2::tpc::UnbinnedResid> *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>));
      static ::ROOT::TGenericClassInfo 
         instance("ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>", -2, "ROOT/RVec.hxx", 1530,
                  typeid(ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR_Dictionary, isa_proxy, 4,
                  sizeof(ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>) );
      instance.SetNew(&new_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR);
      instance.SetNewArray(&newArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR);
      instance.SetDelete(&delete_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR);
      instance.SetDeleteArray(&deleteArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR);
      instance.SetDestructor(&destruct_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< ROOT::VecOps::RVec<o2::tpc::UnbinnedResid> >()));
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>*)
   {
      return GenerateInitInstanceLocal(static_cast<ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal(static_cast<const ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>*>(nullptr))->GetClass();
      ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR_TClassManip(theClass);
   return theClass;
   }

   static void ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(void *p) {
      return  p ? ::new(static_cast<::ROOT::Internal::TOperatorNewHelper*>(p)) ROOT::VecOps::RVec<o2::tpc::UnbinnedResid> : new ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>;
   }
   static void *newArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(Long_t nElements, void *p) {
      return p ? ::new(static_cast<::ROOT::Internal::TOperatorNewHelper*>(p)) ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>[nElements] : new ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>[nElements];
   }
   // Wrapper around operator delete
   static void delete_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(void *p) {
      delete (static_cast<ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>*>(p));
   }
   static void deleteArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(void *p) {
      delete [] (static_cast<ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>*>(p));
   }
   static void destruct_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLUnbinnedResidgR(void *p) {
      typedef ROOT::VecOps::RVec<o2::tpc::UnbinnedResid> current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>

namespace ROOT {
   static TClass *ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR_Dictionary();
   static void ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR_TClassManip(TClass*);
   static void *new_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(void *p = nullptr);
   static void *newArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(Long_t size, void *p);
   static void delete_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(void *p);
   static void deleteArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(void *p);
   static void destruct_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ROOT::VecOps::RVec<o2::tpc::DetInfoResid>*)
   {
      ROOT::VecOps::RVec<o2::tpc::DetInfoResid> *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(ROOT::VecOps::RVec<o2::tpc::DetInfoResid>));
      static ::ROOT::TGenericClassInfo 
         instance("ROOT::VecOps::RVec<o2::tpc::DetInfoResid>", -2, "ROOT/RVec.hxx", 1530,
                  typeid(ROOT::VecOps::RVec<o2::tpc::DetInfoResid>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR_Dictionary, isa_proxy, 4,
                  sizeof(ROOT::VecOps::RVec<o2::tpc::DetInfoResid>) );
      instance.SetNew(&new_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR);
      instance.SetNewArray(&newArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR);
      instance.SetDelete(&delete_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR);
      instance.SetDeleteArray(&deleteArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR);
      instance.SetDestructor(&destruct_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< ROOT::VecOps::RVec<o2::tpc::DetInfoResid> >()));
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ROOT::VecOps::RVec<o2::tpc::DetInfoResid>*)
   {
      return GenerateInitInstanceLocal(static_cast<ROOT::VecOps::RVec<o2::tpc::DetInfoResid>*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ROOT::VecOps::RVec<o2::tpc::DetInfoResid>*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal(static_cast<const ROOT::VecOps::RVec<o2::tpc::DetInfoResid>*>(nullptr))->GetClass();
      ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR_TClassManip(theClass);
   return theClass;
   }

   static void ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(void *p) {
      return  p ? ::new(static_cast<::ROOT::Internal::TOperatorNewHelper*>(p)) ROOT::VecOps::RVec<o2::tpc::DetInfoResid> : new ROOT::VecOps::RVec<o2::tpc::DetInfoResid>;
   }
   static void *newArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(Long_t nElements, void *p) {
      return p ? ::new(static_cast<::ROOT::Internal::TOperatorNewHelper*>(p)) ROOT::VecOps::RVec<o2::tpc::DetInfoResid>[nElements] : new ROOT::VecOps::RVec<o2::tpc::DetInfoResid>[nElements];
   }
   // Wrapper around operator delete
   static void delete_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(void *p) {
      delete (static_cast<ROOT::VecOps::RVec<o2::tpc::DetInfoResid>*>(p));
   }
   static void deleteArray_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(void *p) {
      delete [] (static_cast<ROOT::VecOps::RVec<o2::tpc::DetInfoResid>*>(p));
   }
   static void destruct_ROOTcLcLVecOpscLcLRVeclEo2cLcLtpccLcLDetInfoResidgR(void *p) {
      typedef ROOT::VecOps::RVec<o2::tpc::DetInfoResid> current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ROOT::VecOps::RVec<o2::tpc::DetInfoResid>

namespace ROOT {
   // Registration Schema evolution read functions
   int RecordReadRules_O2ResidualHelpersDict() {
      return 0;
   }
   static int _R__UNIQUE_DICT_(ReadRules_O2ResidualHelpersDict) = RecordReadRules_O2ResidualHelpersDict();R__UseDummy(_R__UNIQUE_DICT_(ReadRules_O2ResidualHelpersDict));
} // namespace ROOT
namespace {
  void TriggerDictionaryInitialization_O2ResidualHelpersDict_Impl() {
    static const char* headers[] = {
"o2_residual_helpers.h",
nullptr
    };
    static const char* includePaths[] = {
"/Users/miranov25/alicesw2/sw/slc9_aarch64/ROOT/v6-36-04-alice9-3/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/O2/dev-local1/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/O2/dev-local1/include/GPU",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/RapidJSON/v1.1.0-alice2-9/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/ONNXRuntime/v1.22.0-12/include/onnxruntime",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/FairMQ/v1.10.0-6/include/fairmq",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/FairMQ/v1.10.0-6/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/fastjet/v3.4.1_1.052-alice3-10/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/JAliEn-ROOT/0.7.15-15/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/ms_gsl/4.2.1-1/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/Common-O2/v1.6.3-21/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/Monitoring/v3.19.8-7/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/libInfoLogger/v2.8.3-5/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/HepMC3/3.3.1-4/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/FairRoot/v18.4.9-alice3-59/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/GEANT3/v4-5-3/include/TGeant3",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/GEANT4_VMC/v6-6-update1-p3-13/include/g4root",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/GEANT4_VMC/v6-6-update1-p3-13/include/geant4vmc",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/VMC/v2-1-4/include/vmc",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/nlohmann_json/v3.11.3-4/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/TBB/v2021.5.0-9/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/Vc/1.4.5-4/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/GSL/v2.8-3/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/FairLogger/v2.3.1-1/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/fmt/11.1.2-5/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/pythia/v8315-alice1-5/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/OpenSSL/v1.1.1m-8/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/boost/v1.83.0-alice2-28/include",
"/Users/miranov25/alicesw2/sw/slc9_aarch64/ROOT/v6-36-04-alice9-3/include/",
"/Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/RDataFrameDSL/ParentChildIndexing/",
nullptr
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "O2ResidualHelpersDict dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
namespace o2{namespace tpc{struct __attribute__((annotate("$clingAutoload$SpacePoints/TrackInterpolation.h")))  __attribute__((annotate("$clingAutoload$o2_residual_helpers.h")))  DetInfoResid;}}
namespace ROOT{namespace VecOps{template <typename T> class __attribute__((annotate(R"ATTRDUMP(__cling__ptrcheck(off))ATTRDUMP"))) __attribute__((annotate("$clingAutoload$ROOT/RVec.hxx")))  __attribute__((annotate("$clingAutoload$o2_residual_helpers.h")))  RVec;
}}
namespace o2{namespace tpc{struct __attribute__((annotate("$clingAutoload$SpacePoints/TrackInterpolation.h")))  __attribute__((annotate("$clingAutoload$o2_residual_helpers.h")))  UnbinnedResid;}}
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "O2ResidualHelpersDict dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "o2_residual_helpers.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"RDataFrameDSL::O2ResidualHelpers::ExtractQMaxTPC", payloadCode, "@",
"RDataFrameDSL::O2ResidualHelpers::ExtractQTotTPC", payloadCode, "@",
"ROOT::VecOps::RVec<o2::tpc::DetInfoResid>", payloadCode, "@",
"ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("O2ResidualHelpersDict",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_O2ResidualHelpersDict_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_O2ResidualHelpersDict_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_O2ResidualHelpersDict() {
  TriggerDictionaryInitialization_O2ResidualHelpersDict_Impl();
}
