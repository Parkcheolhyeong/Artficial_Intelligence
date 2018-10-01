// dib.h : header file
//
// CDIB class
//헤더파일

#ifndef __DIB__
#define __DIB__

class CDIB : public CObject
{
    DECLARE_SERIAL(CDIB)
public:
    CDIB();
    ~CDIB();
    int iBitCount;	//픽셀 당 비트수;
    BITMAPINFO *GetBitmapInfoAddress()
        {return m_pBMI;}                        // ptr to bitmap info
    BYTE *GetBitsAddress()	// 2D 픽셀 포인터를 반환
        {return m_pBits;}                       // ptr to the bits
    RGBQUAD *GetClrTabAddress()
        {return (LPRGBQUAD)(((BYTE *)(m_pBMI)) 
            + sizeof(BITMAPINFOHEADER));}       // ptr to color table
	int CDIB::GetNumClrEntries();
		
    virtual BOOL Create(int width, int height); // create a new DIB
    virtual BOOL Load(CFile *fp);               // load from file
    virtual void Serialize(CArchive& ar);
    virtual void Draw(CDC *pDC, int x, int y, int xWidth, int yWidth);

	BOOL	_CreatePalette();

public:
    BITMAPINFO *m_pBMI;         // pointer to BITMAPINFO struct
    BYTE *m_pBits;  // pointer to the bits
					// 영상에서 2D 픽셀 접근용 포인터
	
	CPalette* m_pPalDib;		// pointer ot the palette;

public:
    int DibHeight() 
        {return m_pBMI->bmiHeader.biHeight;}
    int DibWidth()
        {return m_pBMI->bmiHeader.biWidth;}
        
        	// 2D 픽셀 포인터를 반환 (그레이스케일 & 트루칼라)

};

#endif // __DIB__
