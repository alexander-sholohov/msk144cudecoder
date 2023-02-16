//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: GPLv3
//

#include "f_interop.h"
#include <stdexcept>

#if __GNUC__ > 7
#include <cstddef>
typedef size_t fortran_charlen_t;
#else
typedef int fortran_charlen_t;
#endif

#ifdef USE_EXTERNAL_PBDECODE
#include <Windows.h>
#endif

extern "C"
{
    // --- Fortran routines ---

    void __packjt77_MOD_unpack77(char c77[], // 77
                                 const int* nrx,
                                 char msg[], // 37
                                 int* unpk77_success, fortran_charlen_t, fortran_charlen_t);
}

#ifdef USE_EXTERNAL_PBDECODE

typedef void(__cdecl* packjt77_MOD_unpack77_IMP)(char c77[], // 77
                                                 const int* nrx,
                                                 char msg[], // 37
                                                 int* unpk77_success, fortran_charlen_t, fortran_charlen_t);

packjt77_MOD_unpack77_IMP packjt77_MOD_unpack77_F;

void init_f_interop()
{
    HINSTANCE hinstLib = LoadLibrary(TEXT("bpdecode.dll"));
    if(hinstLib == NULL)
    {
        throw std::runtime_error("Unable to load library bpdecode.dll.");
    }

    packjt77_MOD_unpack77_F = (packjt77_MOD_unpack77_IMP)GetProcAddress(hinstLib, "__packjt77_MOD_unpack77");
    if(packjt77_MOD_unpack77_F == NULL)
    {
        throw std::runtime_error("Wrong proc address __packjt77_MOD_unpack77");
    }
}

#else

void init_f_interop() {}

#endif // USE_EXTERNAL_PBDECODE

void fortran_unpack77(char c77[], // 77
                      const int* nrx,
                      char msg[], // 37
                      int* unpk77_success)
{
#ifdef USE_EXTERNAL_PBDECODE
    packjt77_MOD_unpack77_F(c77, nrx, msg, unpk77_success, 77, 37);
#else
    __packjt77_MOD_unpack77(c77, nrx, msg, unpk77_success, 77, 37);
#endif
}
