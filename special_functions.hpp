/*
 * Special functions header
 */

#ifndef SPECIAL_FUNCTIONS_HPP
#define SPECIAL_FUNCTIONS_HPP

extern "C"
{
    void zwgll_(double*, double*, const int*);
    void dgll_(double*, double*, const double*, const int*, const int*);
    double hgll_(const int*, double*, const double*, const int*);
}

#endif
