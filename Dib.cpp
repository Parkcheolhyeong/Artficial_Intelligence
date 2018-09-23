// dib.cpp : implementation file
//
//

#include "stdafx.h"
#include "dib.h"
#include "malloc.h"
#include "RGBBYTE.h"

#ifdef _DEBUG
#undef THIS_FILE
static char BASED_CODE THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CDIB

IMPLEMENT_SERIAL(CDIB, CObject, 0 /* schema number*/ )

// Create a small DIB here so m_pBMI and m_pBits are always valid
CDIB::CDIB()
{
    m_pBMI = NULL;
    m_pBits = NULL;
    //iBitCount=0; 
	m_pPalDib = NULL;
    Create(0, 0);
}

CDIB::~CDIB()
{
	
   
    if (m_pBMI != NULL) free(m_pBMI);
    if (m_pBits != NULL) free(m_pBits);
   
    //iBitCount =;
	if (m_pPalDib) delete m_pPalDib;
}

/////////////////////////////////////////////////////////////////////////////
// CDIB serialization

// We dont' support this yet  
void CDIB::Serialize(CArchive& ar)
{
    ASSERT(1);
}

/////////////////////////////////////////////////////////////////////////////
// Private functions

static BOOL IsWinDIB(BITMAPINFOHEADER *pBIH)
{
    ASSERT(pBIH);
    if (((BITMAPCOREHEADER *)pBIH)->bcSize == sizeof(BITMAPCOREHEADER)) {
        return FALSE;
    }
    return TRUE;
}

static int NumDIBColorEntries(LPBITMAPINFO pBmpInfo) 
{
    BITMAPINFOHEADER *pBIH;
    BITMAPCOREHEADER *pBCH;
    int iColors,iBitCount;

    ASSERT(pBmpInfo);

    pBIH = &(pBmpInfo->bmiHeader);
    pBCH = (BITMAPCOREHEADER *) pBIH;

    // start off by assuming the color table size from
    // the bit per pixel field
    if (IsWinDIB(pBIH)) {
        iBitCount = pBIH->biBitCount;
    } else {
        iBitCount = pBCH->bcBitCount;
    }

    switch (iBitCount) {
    case 1:
        iColors = 2;
        break;
    case 4:
        iColors = 16;
        break;
    case 8:
        iColors = 256;
        break;
    default:
        iColors = 0;
        break;
    }

    // If this is a Windows DIB, then the color table length
    // is determined by the biClrUsed field if it is non-zero
    if (IsWinDIB(pBIH) && (pBIH->biClrUsed != 0)) {
        iColors = pBIH->biClrUsed;
    }

    return iColors;
}

static int NumDIBColorEntries1(LPBITMAPINFO pBmpInfo) 
{
    BITMAPINFOHEADER *pBIH;
    BITMAPCOREHEADER *pBCH;
    int iBitCount;

    ASSERT(pBmpInfo);

    pBIH = &(pBmpInfo->bmiHeader);
    pBCH = (BITMAPCOREHEADER *) pBIH;

    // start off by assuming the color table size from
    // the bit per pixel field
    if (IsWinDIB(pBIH)) {
        iBitCount = pBIH->biBitCount;
    } else {
        iBitCount = pBCH->bcBitCount;
    }

    return iBitCount;
}


/////////////////////////////////////////////////////////////////////////////
// CDIB commands

// Create a new empty 8bpp DIB with a 256 entry color table
BOOL CDIB::Create(int iWidth, int iHeight)
{
    // delete any existing stuff
    if (m_pBMI != NULL) free(m_pBMI);
    if (m_pBits != NULL) free(m_pBits);

    // allocate memory for the header
    m_pBMI = (BITMAPINFO *) malloc(sizeof(BITMAPINFOHEADER)
                                  + 256 * sizeof(RGBQUAD));
    if (!m_pBMI) {
        TRACE("Out of memory for DIB header");
        return FALSE;
    }

    // allocate memory for the bits (DWORD aligned)
    int iBitsSize = ((iWidth + 3) & ~3) * iHeight;
    m_pBits = (BYTE *)malloc(iBitsSize);



    if (!m_pBits) {
        TRACE("Out of memory for DIB bits");
        free(m_pBMI);
        m_pBMI = NULL;
        return FALSE;
    }
 
  
	// fill in the header info
    BITMAPINFOHEADER *pBI = (BITMAPINFOHEADER *) m_pBMI;
    pBI->biSize = sizeof(BITMAPINFOHEADER);
    pBI->biWidth = iWidth;
    pBI->biHeight = iHeight;
    pBI->biPlanes = 1;
    pBI->biBitCount = 8;
    pBI->biCompression = BI_RGB;
    pBI->biSizeImage = 0;
    pBI->biXPelsPerMeter = 0;
    pBI->biYPelsPerMeter = 0;
    pBI->biClrUsed = 256;
    pBI->biClrImportant = 0;

    // create an arb color table (gray scale)
    RGBQUAD *prgb = GetClrTabAddress();
    for (int i = 0; i < 256; i++) {
        prgb->rgbBlue = prgb->rgbGreen = prgb->rgbRed = (BYTE) i;
        prgb->rgbReserved = 0;
        prgb++;
    }

    // set all the bits to a known state (black)
    memset(m_pBits, 0, iBitsSize);
	
	// create palette;
	_CreatePalette();
    return TRUE;
}


// Load a DIB from an open file.
BOOL CDIB::Load(CFile *fp)
{   int i;
    BOOL bIsPM = FALSE;
    BITMAPINFO *pBmpInfo = NULL;
    BYTE *pBits = NULL;
    
    // get the current file position
    DWORD dwFileStart = fp->GetPosition();

    // read the file header to get the file size and to
    // find where the bits start in the file
    BITMAPFILEHEADER BmpFileHdr;
    int iBytes;
    iBytes = fp->Read(&BmpFileHdr, sizeof(BmpFileHdr));
    if (iBytes != sizeof(BmpFileHdr)) {
        TRACE("Failed to read file header");
        goto $abort;
    }

    // check we have the magic 'BM' at the start
    if (BmpFileHdr.bfType != 0x4D42) {
        TRACE("Not a bitmap file");
        goto $abort;
    }

    // make a wild guess that the file is in Windows DIB
    // format and read the BITMAPINFOHEADER.  If it turns
    // out to be a PM DIB file we'll convert it later.
    BITMAPINFOHEADER BmpInfoHdr;
    iBytes = fp->Read(&BmpInfoHdr, sizeof(BmpInfoHdr)); 
    if (iBytes != sizeof(BmpInfoHdr)) {
        TRACE("Failed to read BITMAPINFOHEADER");
        goto $abort;
    }

    // check we got a real Windows DIB file
    if (BmpInfoHdr.biSize != sizeof(BITMAPINFOHEADER)) {
        if (BmpInfoHdr.biSize != sizeof(BITMAPCOREHEADER)) {
            TRACE(" File is not Windows or PM DIB format");
            goto $abort;
        }

        // set a flag to convert PM file to Win format later
        bIsPM = TRUE;

        // back up the file pointer and read the BITMAPCOREHEADER
        // and create the BITMAPINFOHEADER from it
        fp->Seek(dwFileStart + sizeof(BITMAPFILEHEADER), CFile::begin);
        BITMAPCOREHEADER BmpCoreHdr;
        iBytes = fp->Read(&BmpCoreHdr, sizeof(BmpCoreHdr)); 
        if (iBytes != sizeof(BmpCoreHdr)) {
            TRACE("Failed to read BITMAPCOREHEADER");
            goto $abort;
        }

        BmpInfoHdr.biSize = sizeof(BITMAPINFOHEADER);
        BmpInfoHdr.biWidth = (int) BmpCoreHdr.bcWidth;
        BmpInfoHdr.biHeight = (int) BmpCoreHdr.bcHeight;
        BmpInfoHdr.biPlanes = BmpCoreHdr.bcPlanes;
        BmpInfoHdr.biBitCount = BmpCoreHdr.bcBitCount;
        BmpInfoHdr.biCompression = BI_RGB;
        BmpInfoHdr.biSizeImage = 0;
        BmpInfoHdr.biXPelsPerMeter = 0;
        BmpInfoHdr.biYPelsPerMeter = 0;
        BmpInfoHdr.biClrUsed = 0;
        BmpInfoHdr.biClrImportant = 0;
    }

    // Work out how much memory we need for the BITMAPINFO
    // structure, color table and then for the bits.  
    // Allocate the memory blocks, copy the BmpInfoHdr we have so far 
    // and then read in the color table from the file.
    int iColors;
    int iColorTableSize;
    iColors = NumDIBColorEntries((LPBITMAPINFO) &BmpInfoHdr);
    
	iBitCount =  NumDIBColorEntries1((LPBITMAPINFO) &BmpInfoHdr);
	iColorTableSize = iColors * sizeof(RGBQUAD);
    int iBitsSize;
    int iBISize;
    // always allocate enough room for 256 entries
    iBISize = sizeof(BITMAPINFOHEADER)    
           + 256 * sizeof(RGBQUAD);
    iBitsSize = BmpFileHdr.bfSize - 
                BmpFileHdr.bfOffBits;

    // allocate the memory for the header
    pBmpInfo = (LPBITMAPINFO) malloc(iBISize);
    if (!pBmpInfo) {
        TRACE("Out of memory for DIB header");
        goto $abort;
    }
    memset(pBmpInfo, 0, iBISize);

    // copy the header we already have
    memcpy(pBmpInfo, &BmpInfoHdr, sizeof(BITMAPINFOHEADER));

    // now read the color table in from the file
    if (bIsPM == FALSE) {
        // read the color table from the file
        iBytes = fp->Read(((LPBYTE) pBmpInfo) + sizeof(BITMAPINFOHEADER),
                             iColorTableSize);
        if (iBytes != iColorTableSize) {
            TRACE("Failed to read color table");
            goto $abort;
        }
    } else {
        // read each PM color table entry in turn and convert it
        // to Win DIB format as we go
        LPRGBQUAD lpRGB;
        lpRGB = (LPRGBQUAD) ((LPBYTE) pBmpInfo + sizeof(BITMAPINFOHEADER));
        
        RGBTRIPLE rgbt;
        for (i=0; i<iColors; i++) {
            iBytes = fp->Read(&rgbt, sizeof(RGBTRIPLE));
            if (iBytes != sizeof(RGBTRIPLE)) {
                TRACE("Failed to read RGBTRIPLE");
                goto $abort;
            }
            lpRGB->rgbBlue = rgbt.rgbtBlue;
            lpRGB->rgbGreen = rgbt.rgbtGreen;
            lpRGB->rgbRed = rgbt.rgbtRed;
            lpRGB->rgbReserved = 0;
            lpRGB++;
        }
    }

    // allocate the memory for the bits
    // and read the bits from the file

		
	pBits = (BYTE *) malloc(iBitsSize);
   
	if (!pBits) {
        TRACE("Out of memory for DIB bits");
        goto $abort;
    }


  // seek to the bits in the file
    fp->Seek(dwFileStart + BmpFileHdr.bfOffBits, CFile::begin);

    // read the bits
    
	iBytes = fp->Read(pBits,iBitsSize);
  


    if (iBytes != iBitsSize) {
        TRACE("Failed to read bits");
        goto $abort;
    }

    // Everything went OK
    if (m_pBMI != NULL) free(m_pBMI);
    m_pBMI = pBmpInfo; 
    if (m_pBits != NULL) free (m_pBits);
    m_pBits = pBits;
    
	// create palette;
	_CreatePalette();
    	
	return TRUE;
                
$abort: // something went wrong
    if (pBmpInfo) free(pBmpInfo);
    if (pBits) free (pBits);
    
	return FALSE;    
}


// Draw the DIB to a given DC
void CDIB::Draw(CDC *pDC, int x, int y, int xWidth, int yWidth)
{
    ::StretchDIBits(pDC->GetSafeHdc(),
                  x,                        // dest x
                  y,                        // dest y
                  xWidth, 	                // dest width
                  yWidth,   	            // dest height
                  0,                        // src x
                  0,                        // src y
                  DibWidth(),               // src width
                  DibHeight(),              // src height
                  GetBitsAddress(),         // bits
                  GetBitmapInfoAddress(),   // BITMAPINFO
                  DIB_RGB_COLORS,           // options
                  SRCCOPY);                 // rop
}

// get the number of color table entries
int CDIB::GetNumClrEntries()
{
    return NumDIBColorEntries(m_pBMI);
}

BOOL CDIB::_CreatePalette()
{
    /*****************************************************************
	*
	* CAUTION !!!
	* YOU MUST CALL THIS ROUTINE AFTER LOADING or CREATION()
	*
	******************************************************************/
	DWORD dwColors = NumDIBColorEntries(m_pBMI);
    // Check the DIB has a color table
    if (! dwColors) {
        TRACE("No color table");   
        return FALSE;
    }

    // get a pointer to the RGB quads in the color table
    RGBQUAD * pRGB = GetClrTabAddress();

    // allocate a log pal and fill it with the color table info
    LOGPALETTE *pPal = (LOGPALETTE *) malloc(sizeof(LOGPALETTE) 
                     + dwColors * sizeof(PALETTEENTRY));
    if (!pPal) {
        TRACE("Out of memory for logpal");
        return FALSE;
    }
    pPal->palVersion = 0x300; // Windows 3.0
    pPal->palNumEntries = (WORD) dwColors; // table size
    for (DWORD dw=0; dw<dwColors; dw++) {
        pPal->palPalEntry[dw].peRed = pRGB[dw].rgbRed;
        pPal->palPalEntry[dw].peGreen = pRGB[dw].rgbGreen;
        pPal->palPalEntry[dw].peBlue = pRGB[dw].rgbBlue;
        pPal->palPalEntry[dw].peFlags = 0;
    }

 	if (m_pPalDib) delete m_pPalDib;
	m_pPalDib= new CPalette;
	
	BOOL bResult = m_pPalDib->CreatePalette(pPal);
    free (pPal);
    return bResult;
}