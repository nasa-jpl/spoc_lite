/*****************************************************************************/

/*  
    @file bop.
    @brief Basic algebra operations
*/

/*****************************************************************************/

#ifndef __IBO_BASIC_OPERATIONS2_H__
#define __IBO_BASIC_OPERATIONS2_H__

#include <stdlib.h>
#include <math.h>
#include <complex>
#include <string.h>

#define _USE_MATH_DEFINES
#define BOP_EPSILON (1e-8)
#define TINY 1.0e-20

namespace ibo
{

/* ---------------------------------------- */
/* dst[4] = src1[4] */
template <class T>
inline void bop_copyV4ToV4( T dst[4], T src[4] )
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
}

/* dst[4] = V1[4] + V2[4] */
template <class T>
inline void bop_calcAddV4_V4( T dst[4], T V1[4], T V2[4] )
{
    dst[0] = V1[0] + V2[0];
    dst[1] = V1[1] + V2[1];
    dst[2] = V1[2] + V2[2];
    dst[3] = V1[3] + V2[3];
}

/* dst[4] = V1[4] - V2[4] */
template <class T>
inline void bop_calcSubV4_V4( T dst[4], T V1[4], T V2[4] )
{
    dst[0] = V1[0] - V2[0];
    dst[1] = V1[1] - V2[1];
    dst[2] = V1[2] - V2[2];
    dst[3] = V1[3] - V2[3];
}

/* R[4][4] = identity matrix */
template <class T>
inline void bop_calcIdentityR44( T R[4][4] )
{
    for ( int i=0 ; i<16 ; i++ ) ((T*)R[0])[i] = 0.0;
    R[0][0] = R[1][1] = R[2][2] = R[3][3] = 1.0;
}

/* copy srcR[4][4] to dstR[4][4] */
template <class T>
inline void bop_copyR44ToR44( T dstR[4][4], T srcR[4][4] )
{
    memcpy( &dstR[0][0], &srcR[0][0], sizeof(T)*16 );
}

/*
 * GGemsII, K.Wu, Fast Matrix Inversion 
 */
template <class T>
bool bop_calcInvR44( T dstR[4][4], T srcR[4][4] )
{                          
    bop_copyR44ToR44( dstR, srcR );
        
    int i,j,k;               
    int pvt_i[4], pvt_j[4];            /* Locations of pivot elements */
    T pvt_val;               /* Value of current pivot element */
    T hold;                  /* Temporary storage */
    T determinat;            

    determinat = 1.0;
    for (k=0; k<4; k++)  {
        /* Locate k'th pivot element */
        pvt_val=dstR[k][k];            /* Initialize for search */
        pvt_i[k]=k;
        pvt_j[k]=k;
        for (i=k; i<4; i++) {
            for (j=k; j<4; j++) {
                if (fabs(dstR[i][j]) > fabs(pvt_val)) {
                    pvt_i[k]=i;
                    pvt_j[k]=j;
                    pvt_val=dstR[i][j];
                }
            }
        }

        /* Product of pivots, gives determinant when finished */
        determinat*=pvt_val;
        if (fabs(determinat)<BOP_EPSILON) {    
            return false;  /* Matrix is singular (zero determinant) */
        }

        /* "Interchange" rows (with sign change stuff) */
        i=pvt_i[k];
        if (i!=k) {               /* If rows are different */
            for (j=0; j<4; j++) {
                hold=-dstR[k][j];
                dstR[k][j]=dstR[i][j];
                dstR[i][j]=hold;
            }
        }

        /* "Interchange" columns */
        j=pvt_j[k];
        if (j!=k) {              /* If columns are different */
            for (i=0; i<4; i++) {
                hold=-dstR[i][k];
                dstR[i][k]=dstR[i][j];
                dstR[i][j]=hold;
            }
        }
    
        /* Divide column by minus pivot value */
        for (i=0; i<4; i++) {
            if (i!=k) dstR[i][k]/=( -pvt_val) ; 
        }

        /* Reduce the matrix */
        for (i=0; i<4; i++) {
            hold = dstR[i][k];
            for (j=0; j<4; j++) {
                if (i!=k && j!=k) dstR[i][j]+=hold*dstR[k][j];
            }
        }

        /* Divide row by pivot */
        for (j=0; j<4; j++) {
            if (j!=k) dstR[k][j]/=pvt_val;
        }

        /* Replace pivot by reciprocal (at last we can touch it). */
        dstR[k][k] = 1.0f/pvt_val;
    }

    /* That was most of the work, one final pass of row/column interchange */
    /* to finish */
    for (k=4-2; k>=0; k--) { /* Don't need to work with 1 by 1 corner*/
        i=pvt_j[k];            /* Rows to swap correspond to pivot COLUMN */
        if (i!=k) {            /* If rows are different */
            for(j=0; j<4; j++) {
                hold = dstR[k][j];
                dstR[k][j]=-dstR[i][j];
                dstR[i][j]=hold;
            }
        }

        j=pvt_i[k];           /* Columns to swap correspond to pivot ROW */
        if (j!=k)             /* If columns are different */
            for (i=0; i<4; i++) {
                hold=dstR[i][k];
                dstR[i][k]=-dstR[i][j];
                dstR[i][j]=hold;
            }
    }
    return true;
}

/* dst[4] = R[4][4] x V[4] */
template <class T>
inline void bop_calcR44xV4( T dst[4], T R[4][4], T V[4] )
{
    T tmp[4];
    tmp[0] = R[0][0] * V[0] + R[0][1] * V[1] + R[0][2] * V[2] + R[0][3] * V[3];
    tmp[1] = R[1][0] * V[0] + R[1][1] * V[1] + R[1][2] * V[2] + R[1][3] * V[3];
    tmp[2] = R[2][0] * V[0] + R[2][1] * V[1] + R[2][2] * V[2] + R[2][3] * V[3];
    tmp[3] = R[3][0] * V[0] + R[3][1] * V[1] + R[3][2] * V[2] + R[3][3] * V[3];
    dst[0] = tmp[0];
    dst[1] = tmp[1];
    dst[2] = tmp[2];
    dst[3] = tmp[3];
}

/* dst[3] = R[4][4] x V[3] */
template <class T>
inline void bop_calcR44xV3( T dst[3], T R[4][4], T V[3] )
{
    T tmp[3];
    tmp[0] = R[0][0] * V[0] + R[0][1] * V[1] + R[0][2] * V[2] + R[0][3];
    tmp[1] = R[1][0] * V[0] + R[1][1] * V[1] + R[1][2] * V[2] + R[1][3];
    tmp[2] = R[2][0] * V[0] + R[2][1] * V[1] + R[2][2] * V[2] + R[2][3];
    dst[0] = tmp[0];
    dst[1] = tmp[1];
    dst[2] = tmp[2];
}

/* dst[3] = R[4][4] x V[3], rotation only */
template <class T>
inline void bop_calcR44xV3R( T dst[3], T R[4][4], T V[3] )
{
    T tmp[3];
    tmp[0] = R[0][0] * V[0] + R[0][1] * V[1] + R[0][2] * V[2];
    tmp[1] = R[1][0] * V[0] + R[1][1] * V[1] + R[1][2] * V[2];
    tmp[2] = R[2][0] * V[0] + R[2][1] * V[1] + R[2][2] * V[2];
    dst[0] = tmp[0];
    dst[1] = tmp[1];
    dst[2] = tmp[2];
}

/* Rdst[4][4] = trans R[4][4] */
template <class T>
inline void bop_calcTransR44( T Rdst[4][4], T R[4][4] )
{
    T tmp[4][4];
    copyR44ToR44( tmp, R );
    Rdst[0][0] = tmp[0][0]; Rdst[0][1] = tmp[1][0]; Rdst[0][2] = tmp[2][0]; Rdst[0][3] = tmp[3][0];
    Rdst[1][0] = tmp[0][1]; Rdst[1][1] = tmp[1][1]; Rdst[1][2] = tmp[2][1]; Rdst[1][3] = tmp[3][1];
    Rdst[2][0] = tmp[0][2]; Rdst[2][1] = tmp[1][2]; Rdst[2][2] = tmp[2][2]; Rdst[2][3] = tmp[3][2];
    Rdst[3][0] = tmp[0][3]; Rdst[3][1] = tmp[1][3]; Rdst[3][2] = tmp[2][3]; Rdst[3][3] = tmp[3][3];
}

/* Rdst[4][4] = R1[4][4] x R2[4][4] */
template <class T>
void bop_calcR44xR44( T Rdst[4][4], T R1[4][4], T R2[4][4] )
{
    T tmp[4][4];
    tmp[0][0] = R1[0][0] * R2[0][0] + R1[0][1] * R2[1][0] + R1[0][2] * R2[2][0] + R1[0][3] * R2[3][0];
    tmp[0][1] = R1[0][0] * R2[0][1] + R1[0][1] * R2[1][1] + R1[0][2] * R2[2][1] + R1[0][3] * R2[3][1];
    tmp[0][2] = R1[0][0] * R2[0][2] + R1[0][1] * R2[1][2] + R1[0][2] * R2[2][2] + R1[0][3] * R2[3][2];
    tmp[0][3] = R1[0][0] * R2[0][3] + R1[0][1] * R2[1][3] + R1[0][2] * R2[2][3] + R1[0][3] * R2[3][3];
    tmp[1][0] = R1[1][0] * R2[0][0] + R1[1][1] * R2[1][0] + R1[1][2] * R2[2][0] + R1[1][3] * R2[3][0];
    tmp[1][1] = R1[1][0] * R2[0][1] + R1[1][1] * R2[1][1] + R1[1][2] * R2[2][1] + R1[1][3] * R2[3][1];
    tmp[1][2] = R1[1][0] * R2[0][2] + R1[1][1] * R2[1][2] + R1[1][2] * R2[2][2] + R1[1][3] * R2[3][2];
    tmp[1][3] = R1[1][0] * R2[0][3] + R1[1][1] * R2[1][3] + R1[1][2] * R2[2][3] + R1[1][3] * R2[3][3];
    tmp[2][0] = R1[2][0] * R2[0][0] + R1[2][1] * R2[1][0] + R1[2][2] * R2[2][0] + R1[2][3] * R2[3][0];
    tmp[2][1] = R1[2][0] * R2[0][1] + R1[2][1] * R2[1][1] + R1[2][2] * R2[2][1] + R1[2][3] * R2[3][1];
    tmp[2][2] = R1[2][0] * R2[0][2] + R1[2][1] * R2[1][2] + R1[2][2] * R2[2][2] + R1[2][3] * R2[3][2];
    tmp[2][3] = R1[2][0] * R2[0][3] + R1[2][1] * R2[1][3] + R1[2][2] * R2[2][3] + R1[2][3] * R2[3][3];
    tmp[3][0] = R1[3][0] * R2[0][0] + R1[3][1] * R2[1][0] + R1[3][2] * R2[2][0] + R1[3][3] * R2[3][0];
    tmp[3][1] = R1[3][0] * R2[0][1] + R1[3][1] * R2[1][1] + R1[3][2] * R2[2][1] + R1[3][3] * R2[3][1];
    tmp[3][2] = R1[3][0] * R2[0][2] + R1[3][1] * R2[1][2] + R1[3][2] * R2[2][2] + R1[3][3] * R2[3][2];
    tmp[3][3] = R1[3][0] * R2[0][3] + R1[3][1] * R2[1][3] + R1[3][2] * R2[2][3] + R1[3][3] * R2[3][3];
    T *s = tmp[0], *d = Rdst[0];
    for ( int i=0 ; i<16 ; i++ ) *d++ = *s++;
}

/* dst[4] = R[4][4] x V[4] + T[4] */
template <class T>
inline void bop_calcR44xV4_T4( T dst[4], T R[4][4], T V[4], T t[4] )
{
    calcR44xV4( dst, R, V );
    calcAddV4_V4( dst, dst, t );
}

template <class T>
inline T bop_calcInProdV4_V4( T V1[4], T V2[4] )
{
    return V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2] + V1[3] * V2[3];
}

/* sqrt( (V1[3] - V2[3])^2 ) */
template <class T>
inline T bop_calcSqrtV3_V3( T V1[3], T V2[3] )
{
    return sqrt( (V1[0]-V2[0]) * (V1[0]-V2[0]) +
                 (V1[1]-V2[1]) * (V1[1]-V2[1]) +
                 (V1[2]-V2[2]) * (V1[2]-V2[2]) );
}

template <class T>
inline void bop_calcNormV4( T V[4] ){
    T sum=0.0, inv_sum = sqrt(V[0]*V[0] + V[1]*V[1] + V[2]*V[2] + V[3]*V[3]);
    if ( inv_sum > 0.0 ) sum = 1.0 / inv_sum;
    V[0] *= sum;
    V[1] *= sum;
    V[2] *= sum;
    V[3] *= sum;
}

/* ---------------------------------------- */
/* dst[3] = src1[3] */
template <class T>
inline void bop_copyV3ToV3( T dst[3], T src[3] )
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}

/* dst[3] = R[3][4] x V[3] */
template <class T>
inline void bop_calcR34xV3( T dst[3], T R[3][4], T V[3])
{
    T tmp[3];
    tmp[0] = R[0][0] * V[0] + R[0][1] * V[1] + R[0][2] * V[2];
    tmp[1] = R[1][0] * V[0] + R[1][1] * V[1] + R[1][2] * V[2];
    tmp[2] = R[2][0] * V[0] + R[2][1] * V[1] + R[2][2] * V[2];
    dst[0] = tmp[0];
    dst[1] = tmp[1];
    dst[2] = tmp[2];
}

/* dst[3] = RT[3][4] x V[3] */
template <class T>
inline void bop_calcRT34xV3( T dst[3], T R[3][4], T V[3])
{
    T tmp[3];
    tmp[0] = R[0][0] * V[0] + R[0][1] * V[1] + R[0][2] * V[2] + R[0][3];
    tmp[1] = R[1][0] * V[0] + R[1][1] * V[1] + R[1][2] * V[2] + R[1][3];
    tmp[2] = R[2][0] * V[0] + R[2][1] * V[1] + R[2][2] * V[2] + R[2][3];
    dst[0] = tmp[0];
    dst[1] = tmp[1];
    dst[2] = tmp[2];
}

/* dst[3][4] = R[3][3] x RT[3][4] */
template <class T>
inline void bop_calcR33xRT34( T dst[3][4], T R[3][3], T RT[3][4] )
{
    int i;
    T tmp[3][4];
    
    for ( i=0 ; i<3 ; i++ ){
        tmp[i][0] = R[i][0] * RT[0][0] + R[i][1] * RT[1][0] + R[i][2] * RT[2][0];
        tmp[i][1] = R[i][0] * RT[0][1] + R[i][1] * RT[1][1] + R[i][2] * RT[2][1];
        tmp[i][2] = R[i][0] * RT[0][2] + R[i][1] * RT[1][2] + R[i][2] * RT[2][2];
        tmp[i][3] = R[i][0] * RT[0][3] + R[i][1] * RT[1][3] + R[i][2] * RT[2][3];
    }
    memcpy( dst, tmp, sizeof(T) * 12 );
}

/* ---------------------------------------- */
/* dst[3] = V1[3] + V2[3] */
template <class T>
inline void bop_calcAddV3_V3( T dst[3], const T V1[3], const T V2[3] )
{
    dst[0] = V1[0] + V2[0];
    dst[1] = V1[1] + V2[1];
    dst[2] = V1[2] + V2[2];
}

/* dst[3] = V1[3] - V2[3] */
template <class T>
inline void bop_calcSubV3_V3( T dst[3], const T V1[3], const T V2[3] )
{
    dst[0] = V1[0] - V2[0];
    dst[1] = V1[1] - V2[1];
    dst[2] = V1[2] - V2[2];
}

/* R[3][3] = identity matrix */
template <class T>
inline void bop_calcIdentityR33( T R[3][3] )
{
    for ( int i=0 ; i<9 ; i++ ) ((T*)R[0])[i] = 0.0;
    R[0][0] = R[1][1] = R[2][2] = 1.0;
}

/* copy srcR[3][3] to dstR[3][3] */
template <class T>
inline void bop_copyR33ToR33( T dstR[3][3], T srcR[3][3] )
{
    memcpy( &dstR[0][0], &srcR[0][0], sizeof(T)*9 );
}

#if 0 // 0
/*
 * GGemsII, K.Wu, Fast Matrix Inversion 
 */
template <class T>
inline bool calcInvR33( T dstR[3][3], T srcR[3][3] )
{                          
    copyR33ToR33( dstR, srcR );
        
    int i,j,k;               
    int pvt_i[3], pvt_j[3];            /* Locations of pivot elements */
    T pvt_val;               /* Value of current pivot element */
    T hold;                  /* Temporary storage */
    T determinat;            

    determinat = 1.0;
    for (k=0; k<3; k++)  {
        /* Locate k'th pivot element */
        pvt_val=dstR[k][k];            /* Initialize for search */
        pvt_i[k]=k;
        pvt_j[k]=k;
        for (i=k; i<3; i++) {
            for (j=k; j<3; j++) {
                if (fabs(dstR[i][j]) > fabs(pvt_val)) {
                    pvt_i[k]=i;
                    pvt_j[k]=j;
                    pvt_val=dstR[i][j];
                }
            }
        }

        /* Product of pivots, gives determinant when finished */
        determinat*=pvt_val;
        if (fabs(determinat)<BOPEPSILON) {    
            return false;  /* Matrix is singular (zero determinant) */
        }

        /* "Interchange" rows (with sign change stuff) */
        i=pvt_i[k];
        if (i!=k) {               /* If rows are different */
            for (j=0; j<3; j++) {
                hold=-dstR[k][j];
                dstR[k][j]=dstR[i][j];
                dstR[i][j]=hold;
            }
        }

        /* "Interchange" columns */
        j=pvt_j[k];
        if (j!=k) {              /* If columns are different */
            for (i=0; i<3; i++) {
                hold=-dstR[i][k];
                dstR[i][k]=dstR[i][j];
                dstR[i][j]=hold;
            }
        }
    
        /* Divide column by minus pivot value */
        for (i=0; i<3; i++) {
            if (i!=k) dstR[i][k]/=( -pvt_val) ; 
        }

        /* Reduce the matrix */
        for (i=0; i<3; i++) {
            hold = dstR[i][k];
            for (j=0; j<3; j++) {
                if (i!=k && j!=k) dstR[i][j]+=hold*dstR[k][j];
            }
        }

        /* Divide row by pivot */
        for (j=0; j<3; j++) {
            if (j!=k) dstR[k][j]/=pvt_val;
        }

        /* Replace pivot by reciprocal (at last we can touch it). */
        dstR[k][k] = 1.0f/pvt_val;
    }

    /* That was most of the work, one final pass of row/column interchange */
    /* to finish */
    for (k=3-2; k>=0; k--) { /* Don't need to work with 1 by 1 corner*/
        i=pvt_j[k];            /* Rows to swap correspond to pivot COLUMN */
        if (i!=k) {            /* If rows are different */
            for(j=0; j<3; j++) {
                hold = dstR[k][j];
                dstR[k][j]=-dstR[i][j];
                dstR[i][j]=hold;
            }
        }

        j=pvt_i[k];           /* Columns to swap correspond to pivot ROW */
        if (j!=k)             /* If columns are different */
            for (i=0; i<3; i++) {
                hold=dstR[i][k];
                dstR[i][k]=-dstR[i][j];
                dstR[i][j]=hold;
            }
    }
    return true;
}
#endif

template <class T>
bool bop_calcLuDcmpR33( T mat[3][3], int indx[3] )
{
    T vv[3], big, temp, dum, sum;
    int i, j, k, imax = 0;

    for ( i=0 ; i<3 ; i++ ){
        big = 0.0;
        for ( j=0 ; j<3 ; j++ ){
            if ( (temp = fabs(mat[i][j])) > big )
                big = temp;
        }
        if ( big == 0.0 )
            return false;

        vv[i] = 1.0 / big;
    }
    for ( j=0 ; j<3 ; j++ ){
        for ( i=0 ; i<j ; i++ ){
            sum = mat[i][j];
            for ( k=0 ; k<i ; k++ )
                sum -= mat[i][k] * mat[k][j];
            mat[i][j] = sum;
        }
        big = 0.0;
        for ( i=j; i<3 ; i++ ){
            sum = mat[i][j];
            for ( k=0 ; k<j ; k++ )
                sum -= mat[i][k]*mat[k][j];
            mat[i][j] = sum;
            if ( (dum=vv[i]*fabs(sum)) >= big ){
                big = dum;
                imax = i;
            }
        }
        if ( j != imax ){
            for ( k=0 ; k<3 ; k++ ){
                dum = mat[imax][k];
                mat[imax][k] = mat[j][k];
                mat[j][k] = dum;
            }
            vv[imax] = vv[j];
        }
        indx[j] = imax;
        if ( mat[j][j] == 0.0 ) mat[j][j] = TINY;
        if ( j!=3-1 ){
            dum = 1.0/mat[j][j];
            for ( i=j+1 ; i<3 ; i++ ) mat[i][j] *= dum;
        }
    }
    return true;
}

template <class T>
void bop_calcLuBksbR33( T mat[3][3], int indx[3], T b[3] )
{
    int i,j,ip,ii=-1;
    float sum;

    for ( i=0 ; i<3 ; i++ ){
        ip = indx[i];
        sum = b[ip];
        b[ip] = b[i];
        if ( ii >= 0 )
            for ( j=ii ; j<=i-1 ; j++ ) sum -= mat[i][j]*b[j];
        else if (sum) ii = i;
        b[i] = sum;
    }
    for ( i=3-1 ; i>=0 ; i-- ){
        sum = b[i];
        for ( j=i+1 ; j<3 ; j++ ) sum -= mat[i][j]*b[j];
        b[i] = sum/mat[i][i];
    }
}

/* from NR */
template <class T>
bool bop_calcInvR33( T dst[3][3], T src[3][3] )
{
    int i, j;
    T mat[3][3];

    for ( i=0 ; i<3 ; i++ )
        for ( j=0 ; j<3 ; j++ )
            mat[i][j] = src[i][j];

    int indx[3];
    if ( !bop_calcLuDcmpR33( mat, indx ) ) return false;
    
    T col[3];
    for ( i=0 ; i<3 ; i++ ){
        for ( j=0 ; j<3 ; j++ ) col[j] = 0.0;
        col[i] = 1.0;
        bop_calcLuBksbR33( mat, indx, col );
        for ( j=0 ; j<3 ; j++ ) dst[j][i] = col[j];
    }
    return true;
}

/* dst[3] = R[3][3] x V[3] */
template <class T>
inline void bop_calcR33xV3( T dst[3], T R[3][3], const T V[3] )
{
    T tmp[3];
    tmp[0] = R[0][0] * V[0] + R[0][1] * V[1] + R[0][2] * V[2];
    tmp[1] = R[1][0] * V[0] + R[1][1] * V[1] + R[1][2] * V[2];
    tmp[2] = R[2][0] * V[0] + R[2][1] * V[1] + R[2][2] * V[2];
    dst[0] = tmp[0];
    dst[1] = tmp[1];
    dst[2] = tmp[2];
}

/* Rdst[3][3] = trans R[3][3] */
template <class T>
inline void bop_calcTransR33( T Rdst[3][3], T R[3][3] )
{
    T tmp[3][3];
    bop_copyR33ToR33( tmp, R );
    Rdst[0][0] = tmp[0][0]; Rdst[0][1] = tmp[1][0]; Rdst[0][2] = tmp[2][0];
    Rdst[1][0] = tmp[0][1]; Rdst[1][1] = tmp[1][1]; Rdst[1][2] = tmp[2][1];
    Rdst[2][0] = tmp[0][2]; Rdst[2][1] = tmp[1][2]; Rdst[2][2] = tmp[2][2];
}

/* Rdst[3][3] = R1[3][3] x R2[3][3] */
template <class T>
inline void bop_calcR33xR33( T Rdst[3][3], T R1[3][3], T R2[3][3] )
{
    T tmp[3][3];
    tmp[0][0] = R1[0][0] * R2[0][0] + R1[0][1] * R2[1][0] + R1[0][2] * R2[2][0];
    tmp[0][1] = R1[0][0] * R2[0][1] + R1[0][1] * R2[1][1] + R1[0][2] * R2[2][1];
    tmp[0][2] = R1[0][0] * R2[0][2] + R1[0][1] * R2[1][2] + R1[0][2] * R2[2][2];
    tmp[1][0] = R1[1][0] * R2[0][0] + R1[1][1] * R2[1][0] + R1[1][2] * R2[2][0];
    tmp[1][1] = R1[1][0] * R2[0][1] + R1[1][1] * R2[1][1] + R1[1][2] * R2[2][1];
    tmp[1][2] = R1[1][0] * R2[0][2] + R1[1][1] * R2[1][2] + R1[1][2] * R2[2][2];
    tmp[2][0] = R1[2][0] * R2[0][0] + R1[2][1] * R2[1][0] + R1[2][2] * R2[2][0];
    tmp[2][1] = R1[2][0] * R2[0][1] + R1[2][1] * R2[1][1] + R1[2][2] * R2[2][1];
    tmp[2][2] = R1[2][0] * R2[0][2] + R1[2][1] * R2[1][2] + R1[2][2] * R2[2][2];
    T *s = tmp[0], *d = Rdst[0];
    for ( int i=0 ; i<9 ; i++ ) *d++ = *s++;
}

/* dst[3] = R[3][3] x V[3] + T[3] */
template <class T>
inline void bop_calcR33xV3_T3( T dst[3], T R[3][3], const T V[3], const T t[3] ){
    bop_calcR33xV3( dst, R, V );
    bop_calcAddV3_V3( dst, dst, t );
}

template <class T>
inline T bop_calcInProdV3_V3( const T V1[3], const T V2[3] )
{
    return V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2];
}

template <class T>
inline void bop_calcOutProdV3_V3( T dst[3], const T V1[3], const T V2[3] )
{
    T tmp[3];
    tmp[0] = V1[1] * V2[2] - V1[2] * V2[1]; 
    tmp[1] = V1[2] * V2[0] - V1[0] * V2[2];
    tmp[2] = V1[0] * V2[1] - V1[1] * V2[0];
    dst[0] = tmp[0];
    dst[1] = tmp[1];
    dst[2] = tmp[2];
}

template <class T>
inline void bop_calcNormV3( T V[3] ){
    T sum=0.0, inv_sum = sqrt(V[0]*V[0] + V[1]*V[1] + V[2]*V[2]);
    if ( inv_sum > 0.0 ) sum = 1.0 / inv_sum;
    V[0] *= sum;
    V[1] *= sum;
    V[2] *= sum;
}

/* ---------------------------------------- */
/* sqrt( (V1[2] - V2[2])^2 ) */
template <class T>
inline T bop_calcSqrtV2_V2( T V1[2], T V2[2] )
{
    return sqrt( (V1[0]-V2[0]) * (V1[0]-V2[0]) +
                 (V1[1]-V2[1]) * (V1[1]-V2[1]) );
}

template <class T>
inline T bop_calcInProdV2_V2( T V1[2], T V2[2] )
{
    return V1[0] * V2[0] + V1[1] * V2[1];
}

template <class T>
inline void bop_calcNormV2( T V[2] ){
    T sum=0.0, inv_sum = sqrt(V[0]*V[0] + V[1]*V[1]);
    if ( inv_sum > 0.0 ) sum = 1.0 / inv_sum;
    V[0] *= sum;
    V[1] *= sum;
}

/* ---------------------------------------- */
/* Quaternion */

/* q[4] = identity quaternion */
template <class T>
inline void bop_calcIdentityQ4( T q[4] )
{
    q[0] = 0.0;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 1.0;
}

template <class T>
void bop_convQ4ToA4( T dst[4], const T src[4] )
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];

    double angle = src[3];
    if      ( angle>1.0 )  angle = 1.0;
    else if ( angle<-1.0 ) angle = -1.0;
    dst[3] = acos( angle ) * 2.0;
  
    if ( (dst[0] == 0.0) && (dst[1] == 0.0) && (dst[2] == 0.0) ){
        dst[0] = 0; dst[1] = 0; dst[2] = 1.0;
    }
    calcNormV3( dst );

    while ( dst[3] > M_PI ) dst[3] -= M_PI + M_PI;
}

template <class T>
inline void bop_convA4ToQ4( T dst[4], const T src[4] )
{
    T src2[3];
    T t=src[3]*0.5;
    T s=sin(t), c=cos(t);

    src2[0] = src[0];
    src2[1] = src[1];
    src2[2] = src[2];
    calcNormV3( src2 );

    dst[0] = s * src2[0];
    dst[1] = s * src2[1];
    dst[2] = s * src2[2];
    dst[3] = c;
}

template <class T>
inline void bop_convAP7ToQP7( T dst[7], const T src[7] )
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    bop_convA4ToQ4<T>( &dst[3], &src[3] );
}

template <class T>
inline void bop_convQP7ToAP7( T dst[7], const T src[7] )
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    convQ4ToA4( &dst[3], &src[3] );
}

template <class T>
void bop_convQ4ToR33( T dst[3][3], const T src[4] )
{
    T s,xs,ys,zs,wx,wy,wz,xx,xy,xz,yy,yz,zz;

    T l = src[0]*src[0] + src[1]*src[1] + src[2]*src[2] + src[3]*src[3];
    if ( l < BOP_EPSILON && l > -BOP_EPSILON )
        s = 1.0;
    else
        s = 2.0/l;

    xs = src[0] * s;   ys = src[1] * s;  zs = src[2] * s;
    wx = src[3] * xs;  wy = src[3] * ys; wz = src[3] * zs;
    xx = src[0] * xs;  xy = src[0] * ys; xz = src[0] * zs;
    yy = src[1] * ys;  yz = src[1] * zs; zz = src[2] * zs;

    dst[0][0]=1.0 - (yy +zz);
    dst[0][1]=xy - wz;
    dst[0][2]=xz + wy;
    dst[1][0]=xy + wz;
    dst[1][1]=1.0 - (xx +zz);
    dst[1][2]=yz - wx;
    dst[2][0]=xz - wy;
    dst[2][1]=yz + wx;
    dst[2][2]=1.0 - (xx + yy);
    //dst[3][0]=dst[3][1]=dst[3][2]=dst[0][3]=dst[1][3]=dst[2][3]=0.0;
    //dst[3][3]=1.0;

#if 0    
    float coef = 1.0/(src[0]*src[0] + src[1]*src[1] + src[2]*src[2] + src[3]*src[3]);
    
    dst[0][0] = (src[3]*src[3] + src[0]*src[0] - src[1]*src[1] - src[2]*src[2])*coef;  
    dst[0][1] = (2.0*(src[0]*src[1] - src[3]*src[2]))*coef;  
    dst[0][2] = (2.0*(src[0]*src[2] + src[3]*src[1]))*coef;  
    dst[1][0] = (2.0*(src[0]*src[1] + src[3]*src[2]))*coef;  
    dst[1][1] = (src[3]*src[3] - src[0]*src[0] + src[1]*src[1] - src[2]*src[2])*coef;  
    dst[1][2] = (2.0*(src[1]*src[2] - src[3]*src[0]))*coef;  
    dst[2][0] = (2.0*(src[0]*src[2] - src[3]*src[1]))*coef;  
    dst[2][1] = (2.0*(src[1]*src[2] + src[3]*src[0]))*coef;  
    dst[2][2] = (src[3]*src[3] - src[0]*src[0] - src[1]*src[1] + src[2]*src[2])*coef;
#endif    
}

template <class T>
inline void bop_convR33ToQ4( T dst[4], T src[3][3] )
{
    T t, s;

    t = src[0][0] + src[1][1] + src[2][2] + 1;
    if( t >= 1.0e-6 ){
        s = sqrt( t ) * 2.0;
        dst[0] = ( src[2][1] - src[1][2] ) / s;
        dst[1] = ( src[0][2] - src[2][0] ) / s;
        dst[2] = ( src[1][0] - src[0][1] ) / s;
        dst[3] = 0.25 * s;
    }
    else if ( src[0][0] >= src[1][1] && src[0][0] >= src[2][2] )
    {
        s = sqrt( 1.0 + src[0][0] - src[1][1] - src[2][2] ) * 2;
        dst[0] = 0.25 * s;
        dst[1] = ( src[0][1] + src[1][0] ) / s;
        dst[2] = ( src[0][2] + src[2][0] ) / s;
        dst[3] = ( src[1][2] - src[2][1] ) / s;
    }
    else if ( src[1][1] >= src[0][0] && src[1][1] >= src[2][2] )
    {
        s = sqrt( 1.0 + src[1][1] - src[0][0] - src[2][2]) * 2;
        dst[0] = ( src[0][1] + src[1][0] ) / s;
        dst[1] = 0.25 * s;
        dst[2] = ( src[1][2] + src[2][1] ) / s;
        dst[3] = ( src[0][2] - src[2][0] ) / s;
    }
    else{
        s = sqrt( 1.0 + src[2][2] - src[0][0] - src[1][1] ) * 2;
        dst[0] = ( src[0][2] + src[2][0] ) / s;
        dst[1] = ( src[1][2] + src[2][1] ) / s;
        dst[2] = 0.25 * s;
        dst[3] = ( src[0][1] - src[1][0] ) / s;
    }
}

template <class T>
void bop_convA4ToR33( T dst[3][3], const T src[4] )
{
	T st,ct,u,v,w,coef;
    
	st = sin(src[3]);
	ct = cos(src[3]);
	coef = 1.0/sqrt( src[0]*src[0] + src[1]*src[1] + src[2]*src[2] );
	u=src[0]*coef;
	v=src[1]*coef;
	w=src[2]*coef;
	dst[0][0]=u*u+(1.0-u*u)*ct;
	dst[0][1]=u*v*(1.0-ct)-w*st;
	dst[0][2]=u*w*(1.0-ct)+v*st;
	dst[1][0]=u*v*(1.0-ct)+w*st;
	dst[1][1]=v*v+(1.0-v*v)*ct;
	dst[1][2]=v*w*(1.0-ct)-u*st;
	dst[2][0]=u*w*(1.0-ct)-v*st;
	dst[2][1]=v*w*(1.0-ct)+u*st;
	dst[2][2]=w*w+(1.0-w*w)*ct;
}

template <class T>
inline void bop_convQP7ToR34( T dst[3][4], const T src[7] )
{
    float R[3][3];
    bop_convQ4ToR33( R, &src[3] );
    
    for ( int i=0 ; i<3 ; i++ ){
        dst[i][0] = R[i][0];
        dst[i][1] = R[i][1];
        dst[i][2] = R[i][2];
        dst[i][3] = src[i];
    }
}

template <class T>
void bop_calcQ4xQ4( T dst[4], const T Q1[4], const T Q2[4] )
{
    T tmp[4];
    
    tmp[0] = Q1[3] * Q2[0] + Q2[3] * Q1[0] + Q1[1] * Q2[2] - Q1[2] * Q2[1];
    tmp[1] = Q1[3] * Q2[1] + Q2[3] * Q1[1] + Q1[2] * Q2[0] - Q1[0] * Q2[2];
    tmp[2] = Q1[3] * Q2[2] + Q2[3] * Q1[2] + Q1[0] * Q2[1] - Q1[1] * Q2[0];
    tmp[3] = Q1[3] * Q2[3] - Q1[0] * Q2[0] - Q1[1] * Q2[1] - Q1[2] * Q2[2];
    calcNormV4(tmp);

    if ( tmp[3] < 0.0 ){
        dst[0] = -tmp[0];
        dst[1] = -tmp[1];
        dst[2] = -tmp[2];
        dst[3] = -tmp[3];
    }
    else{
        dst[0] = tmp[0];
        dst[1] = tmp[1];
        dst[2] = tmp[2];
        dst[3] = tmp[3];
    }
}

template <class T>
inline void bop_calcInvQ4( T dst[4], const T src[4] )
{
    dst[0] = -src[0];
    dst[1] = -src[1];
    dst[2] = -src[2];
    dst[3] = src[3];
}

/* dst = Q1 x Q2 */
template <class T>
inline void bop_calcQP7xQP7( T dst[7], const T Q1[7], const T Q2[7] )
{
    T R[3][3];
    
    bop_convQ4ToR33( R, &Q1[3] );
    bop_calcR33xV3( dst, R, Q2 );
    bop_calcAddV3_V3( dst, dst, Q1 );

    bop_calcQ4xQ4( &dst[3], &Q1[3], &Q2[3] );
}

/* dst = inv(Q1) x Q2 */
template <class T>
inline void bop_calcInvQP7xQP7( T dst[7], const T Q1[7], const T Q2[7] )
{
    T inv_q[4];
    T R[3][3];
    
    bop_calcInvQ4( inv_q, &Q1[3] );

    bop_calcSubV3_V3( dst, Q2, Q1 );
    bop_convQ4ToR33( R, inv_q );
    bop_calcR33xV3( dst, R, dst );

    bop_calcQ4xQ4( &dst[3], inv_q, &Q2[3] );
}

/* dst = (1-t) * Q1 + t * Q2 */
template <class T>
void bop_calcSlerpQ4_Q4( T dst[4], const T Q1[4], const T Q2[4], const T t )
{
    double theta = acos( bop_calcInProdV4_V4( Q1, Q2 ) );
    for ( int i=0 ; i<4 ; i++ )
        dst[i] = (sin((1-t)*theta) * Q1[i] + sin(t*theta) * Q2[i]) / sin(theta);
}

template <class T>
void bop_convRPYToQ4( T dst[4], T src[3] )
{
    T a[4], q_roll[4], q_pitch[4], q_yaw[4], q[4];

    a[0] = 1.0;
    a[1] = 0.0;
    a[2] = 0.0;
    a[3] = src[2];
    bop_convA4ToQ4( q_yaw, a );

    a[0] = 0.0;
    a[1] = 1.0;
    a[2] = 0.0;
    a[3] = src[1];
    bop_convA4ToQ4( q_pitch, a );

    a[0] = 0.0;
    a[1] = 0.0;
    a[2] = 1.0;
    a[3] = src[0];
    bop_convA4ToQ4( q_roll, a );
    
    bop_calcQ4xQ4( q, q_pitch, q_yaw );
    bop_calcQ4xQ4( dst, q_roll, q );
}

template <class T>
void bop_convR33ToRPY( T rpy[3], T R[3][3] )
{
    /* rpy[0] = atan2(R[1][0], R[0][0]); */
    /* rpy[1] = atan2 (-R[2][0], (cos(rpy[0]) * R[0][0]) + (sin(rpy[0]) * R[1][0])); */
    /* rpy[2] = atan2 ((sin(rpy[0]) * R[0][2]) - (cos(rpy[0]) * R[1][2]), */
    /*                 (cos(rpy[0]) * R[1][1]) - (sin(rpy[0]) * R[0][1])); */
    rpy[2] = atan2 (R[1][0], R[0][0]);
    rpy[1] = atan2 (-R[2][0], (cos(rpy[2]) * R[0][0]) + (sin(rpy[2]) * R[1][0]));
    rpy[0] = atan2 ((sin(rpy[2]) * R[0][2]) - (cos(rpy[2]) * R[1][2]),
                    (cos(rpy[2]) * R[1][1]) - (sin(rpy[2]) * R[0][1]));
}

template <class T>
void bop_convQ4ToRPY( T rpy[3], T src[4] )
{
    T R[3][3];
    bop_convQ4ToR33( R, src );
    bop_convR33ToRPY( rpy, R );
}

// Box-Muller method
template <class T>
T bop_calcGaussianNoise( T mean, T sigma )
{
	T rsq,v1,v2;

    do {
        v1 = 2.0*(T)rand()/(RAND_MAX+1.0) - 1.0;
        v2 = 2.0*(T)rand()/(RAND_MAX+1.0) - 1.0;
        rsq = v1*v1 + v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);

    return v1 * sqrt(-2.0*log(rsq)/rsq) * sigma + mean;
}

/* ---------------------------------------- */
/* Analytical Solution to 4th, 3rd, 2nd Order Polynomial */

// analytical solution of x^2 + ce[0] x + ce[1] = 0
template <class T>
void bop_solveQuadricPolynomial(std::complex<T> ret[2], T ce[2])
{
    std::complex<double> a(ce[0]*ce[0] - 4.0*ce[1], 0.0);
    a = sqrt(a);

    ret[0] = (-ce[0] + a)*0.5;
    ret[1] = (-ce[0] - a)*0.5;
 }

// analytical solution of x^3 + ce[0] x^2 + ce[1] x + ce[2] = 0
template <class T>
void bop_solveCubicPolynomial(std::complex<T> ret[3], T ce[3])
{
    // y^3 + e y + f = 0 (x = y - ce[0]/3)
    double e = -ce[0]*ce[0]/3.0 + ce[1];
    double f = ce[0]*ce[0]*ce[0]/13.5 - ce[0]*ce[1]/3.0 + ce[2];

    // z^6 + f z^3 - e^3/27 = 0 (y = z - e/(3z))
    // u^2 + f u   - e^3/27 = 0 (u = z^3)
    //   u = -f/2 +- sqrt(f*f/4.0+e*e*e/27.0)
    double g = f*f/4.0+e*e*e/27.0;
    if (g < 0.0){
        // u = -f/2 +- i sqrt(-g)
        // z = cbrt(|u|) exp(i(atan2(sqrt(-g), -f/2) + 2*M_PI*i)/3)
        //   = a + i b
        double len = pow(-e*e*e/27.0, 1.0/6.0); // cbrt(|u|)
        double theta = atan2(sqrt(-g), -f/2);
        // y becomes a real number, the imagenary part is alwayz zero
        for (int i=0 ; i<3 ; i++){
            double a = len * cos((theta + 2.0*M_PI*i)/3.0);
            double b = len * sin((theta + 2.0*M_PI*i)/3.0);
            ret[i] = a*((a*a+b*b) - e/3.0)/(a*a+b*b) - ce[0]/3;
        }
    }
    else{
        g = -f/2+sqrt(g);
        
        double z;
        if (g < 0.0)
            z = -pow(-g, 1.0/3.0);
        else
            z = pow(g, 1.0/3.0);
        ret[0] = z - e/3.0/z;
        
        double ce2[2];
        ce2[0] = ret[0].real();
        ce2[1] = e + ce2[0]*ret[0].real();
        solveQuadricPolynomial(ret+1, ce2);

        ret[0] -= ce[0]/3.0;
        ret[1] -= ce[0]/3.0;
        ret[2] -= ce[0]/3.0;
    }
}

// analytical solution of x^4 + ce[0] x^3 + ce[1] x^2 + ce[2] x + ce[3] = 0
template <class T>
void bop_solveQuarticPolynomial(std::complex<T> ret[4], T ce[4])
{
    // y^4 + e y^2 + f y + g = 0 (x = y - ce[0]/4)
    double e = -3.0*ce[0]*ce[0]/8.0 + ce[1];
    double f = ce[0]*ce[0]*ce[0]/8.0 - ce[0]*ce[1]/2.0 + ce[2];
    double g = -3.0*ce[0]*ce[0]*ce[0]*ce[0]/256.0 + ce[0]*ce[0]*ce[1]/16.0 - ce[0]*ce[2]/4.0 + ce[3];
    
    double ce2[3];
    
    if (g == 0.0){
        ret[0] = 0.0;

        ce2[0] = 0.0;
        ce2[1] = e;
        ce2[2] = f;
        bop_solveCubicPolynomial(ret+1, ce2);
    }
    else if (f == 0.0){
        ce2[0] = e;
        ce2[1] = g;
        std::complex<double> ret2[2];
        bop_solveQuadricPolynomial(ret2, ce2);
        
        ret[0] = sqrt(ret2[0]);
        ret[1] = -ret[0];
        ret[2] = sqrt(ret2[1]);
        ret[3] = -ret[2];
    }
    else{
        // z^3 + (e/2) z^2 + ((e^2-4g)/16) z - f^2/64 = 0
        ce2[0] = e/2;
        ce2[1] = (e*e-4.0*g)/16.0;
        ce2[2] = -f*f/64.0;
        std::complex<double> ret2[3];
        bop_solveCubicPolynomial(ret2, ce2);
        
        std::complex<double> p, q, r;
        p = sqrt(ret2[0]);
        q = sqrt(ret2[1]);
        r = -f/(8.0*p*q);

        ret[0] =  p + q + r;
        ret[1] =  p - q - r;
        ret[2] = -p + q - r;
        ret[3] = -p - q + r;
    }

    for (int i=0 ; i<4 ; i++) ret[i] -= ce[0]/4.0;
}

} // namespace ibo

#endif // __BASIC_OPERATIONS2_H__
