//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

void fortran_unpack77(char c77[], // 77
                      const int* nrx,
                      char msg[], // 37
                      int* unpk77_success);

void init_f_interop();
