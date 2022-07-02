# ECG_detect_OSA

------------------------------------------
This program is designed to recognize **Obstructive sleep apnea**. Using I²C connect MAX86150 with ARC EM9D board GMA303KU on board and Convolutional Neural Networks to recognize these diease.

<img width="450" alt="PC2" src="https://user-images.githubusercontent.com/87894572/176604629-1e02e3cf-c78f-47bf-a87c-b511c30e8c5f.png">

* [Introduction](#introduction)
* [Hardware and Software Setup](#hardware-and-software-setup)
	* [Required Hardware](#required-hardware)
	* [Required Software](#required-software)
	* [Hardware Connection](#hardware-connection)
* [User Manual](#user-manual)
	* [Compatible Model](#compatible-model)
	* [C work on board](#c-work-on-board)
		* [Initial each device](#initial-each-device)
	* [Python work for deep learning](#python-work-for-deep-learning)
		* [Loading dataset and pre-processing](#loading-dataset-and-pre-processing)
		* [Setup model](#setup-model)
	


## Introduction
- 1 We use 3-axis accelerometer to determine user whether is go to sleep.
- 2 Using I²C to trasport ECG data to ARC EM9D board.
- 3 Using Convolutional Neural Networks model to recognize real time ECG data.
- 4 Show the result to user.

All hardware are in the picture following:

<img width="450" alt="PC2" src="https://user-images.githubusercontent.com/87894572/176442424-26c242db-f6ff-4690-a84b-176652868726.png">

## Hardware and Software Setup
### Required Hardware
-  ARC EM9D board
-  MAX86150
-  OLED1306

All hardware are in the picture following:

<img width="450" alt="PC2" src="https://user-images.githubusercontent.com/87894572/176450960-a8cc7ce9-5fe4-49f3-83ad-ed625e810ab7.png">

### Required Software
- Metaware or ARC GNU Toolset
- Serial port terminal, such as putty, tera-term or minicom
- VirtualBox(Ubuntu 20.04)
- Cygwin64 Terminal

### Hardware Connection
- ARC EM9D, MAX861150 and OLED1306 connected by wire.

## User Manual

### Compatible Model

1. Download [Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/)

1. Give the **Apnea-ECG Database** which held Computers contest with Cardiology in 2000.

1. Use **Tensorflow 2.x** and **Python 3.7 up** to training model

1. Import module to read Apnea-ECG Database

```python
import wfdb
```
### C work on board
#### OLED1306 part

- Open "synopsys_i2c_oled1306.c" and copy code

```c
#include "synopsys_i2c_oled1306.h"

#define USE_SS_IIC_X USE_SS_IIC_0

const uint8_t OledFontTable[] =
{
        0x00, 0x00, 0x00, 0x00, 0x00,   // space
        0x00, 0x00, 0x2f, 0x00, 0x00,   // !
        0x00, 0x07, 0x00, 0x07, 0x00,   // "
        0x14, 0x7f, 0x14, 0x7f, 0x14,   // #
        0x24, 0x2a, 0x7f, 0x2a, 0x12,   // $
        0x23, 0x13, 0x08, 0x64, 0x62,   // %
        0x36, 0x49, 0x55, 0x22, 0x50,   // &
        0x00, 0x05, 0x03, 0x00, 0x00,   // '
        0x00, 0x1c, 0x22, 0x41, 0x00,   // (
        0x00, 0x41, 0x22, 0x1c, 0x00,   // )
        0x14, 0x08, 0x3E, 0x08, 0x14,   // *
        0x08, 0x08, 0x3E, 0x08, 0x08,   // +
        0x00, 0x00, 0xA0, 0x60, 0x00,   // ,
        0x08, 0x08, 0x08, 0x08, 0x08,   // -
        0x00, 0x60, 0x60, 0x00, 0x00,   // .
        0x20, 0x10, 0x08, 0x04, 0x02,   // /

        0x3E, 0x51, 0x49, 0x45, 0x3E,   // 0
        0x00, 0x42, 0x7F, 0x40, 0x00,   // 1
        0x42, 0x61, 0x51, 0x49, 0x46,   // 2
        0x21, 0x41, 0x45, 0x4B, 0x31,   // 3
        0x18, 0x14, 0x12, 0x7F, 0x10,   // 4
        0x27, 0x45, 0x45, 0x45, 0x39,   // 5
        0x3C, 0x4A, 0x49, 0x49, 0x30,   // 6
        0x01, 0x71, 0x09, 0x05, 0x03,   // 7
        0x36, 0x49, 0x49, 0x49, 0x36,   // 8
        0x06, 0x49, 0x49, 0x29, 0x1E,   // 9

        0x00, 0x36, 0x36, 0x00, 0x00,   // :
        0x00, 0x56, 0x36, 0x00, 0x00,   // ;
        0x08, 0x14, 0x22, 0x41, 0x00,   // <
        0x14, 0x14, 0x14, 0x14, 0x14,   // =
        0x00, 0x41, 0x22, 0x14, 0x08,   // >
        0x02, 0x01, 0x51, 0x09, 0x06,   // ?
        0x32, 0x49, 0x59, 0x51, 0x3E,   // @

        0x7C, 0x12, 0x11, 0x12, 0x7C,   // A
        0x7F, 0x49, 0x49, 0x49, 0x36,   // B
        0x3E, 0x41, 0x41, 0x41, 0x22,   // C
        0x7F, 0x41, 0x41, 0x22, 0x1C,   // D
        0x7F, 0x49, 0x49, 0x49, 0x41,   // E
        0x7F, 0x09, 0x09, 0x09, 0x01,   // F
        0x3E, 0x41, 0x49, 0x49, 0x7A,   // G
        0x7F, 0x08, 0x08, 0x08, 0x7F,   // H
        0x00, 0x41, 0x7F, 0x41, 0x00,   // I
        0x20, 0x40, 0x41, 0x3F, 0x01,   // J
        0x7F, 0x08, 0x14, 0x22, 0x41,   // K
        0x7F, 0x40, 0x40, 0x40, 0x40,   // L
        0x7F, 0x02, 0x0C, 0x02, 0x7F,   // M
        0x7F, 0x04, 0x08, 0x10, 0x7F,   // N
        0x3E, 0x41, 0x41, 0x41, 0x3E,   // O
        0x7F, 0x09, 0x09, 0x09, 0x06,   // P
        0x3E, 0x41, 0x51, 0x21, 0x5E,   // Q
        0x7F, 0x09, 0x19, 0x29, 0x46,   // R
        0x46, 0x49, 0x49, 0x49, 0x31,   // S
        0x01, 0x01, 0x7F, 0x01, 0x01,   // T
        0x3F, 0x40, 0x40, 0x40, 0x3F,   // U
        0x1F, 0x20, 0x40, 0x20, 0x1F,   // V
        0x3F, 0x40, 0x38, 0x40, 0x3F,   // W
        0x63, 0x14, 0x08, 0x14, 0x63,   // X
        0x07, 0x08, 0x70, 0x08, 0x07,   // Y
        0x61, 0x51, 0x49, 0x45, 0x43,   // Z

        0x00, 0x7F, 0x41, 0x41, 0x00,   // [
        0x55, 0xAA, 0x55, 0xAA, 0x55,   // Backslash (Checker pattern)
        0x00, 0x41, 0x41, 0x7F, 0x00,   // ]
        0x04, 0x02, 0x01, 0x02, 0x04,   // ^
        0x40, 0x40, 0x40, 0x40, 0x40,   // _
        0x00, 0x03, 0x05, 0x00, 0x00,   // `

        0x20, 0x54, 0x54, 0x54, 0x78,   // a
        0x7F, 0x48, 0x44, 0x44, 0x38,   // b
        0x38, 0x44, 0x44, 0x44, 0x20,   // c
        0x38, 0x44, 0x44, 0x48, 0x7F,   // d
        0x38, 0x54, 0x54, 0x54, 0x18,   // e
        0x08, 0x7E, 0x09, 0x01, 0x02,   // f
        0x18, 0xA4, 0xA4, 0xA4, 0x7C,   // g
        0x7F, 0x08, 0x04, 0x04, 0x78,   // h
        0x00, 0x44, 0x7D, 0x40, 0x00,   // i
        0x40, 0x80, 0x84, 0x7D, 0x00,   // j
        0x7F, 0x10, 0x28, 0x44, 0x00,   // k
        0x00, 0x41, 0x7F, 0x40, 0x00,   // l
        0x7C, 0x04, 0x18, 0x04, 0x78,   // m
        0x7C, 0x08, 0x04, 0x04, 0x78,   // n
        0x38, 0x44, 0x44, 0x44, 0x38,   // o
        0xFC, 0x24, 0x24, 0x24, 0x18,   // p
        0x18, 0x24, 0x24, 0x18, 0xFC,   // q
        0x7C, 0x08, 0x04, 0x04, 0x08,   // r
        0x48, 0x54, 0x54, 0x54, 0x20,   // s
        0x04, 0x3F, 0x44, 0x40, 0x20,   // t
        0x3C, 0x40, 0x40, 0x20, 0x7C,   // u
        0x1C, 0x20, 0x40, 0x20, 0x1C,   // v
        0x3C, 0x40, 0x30, 0x40, 0x3C,   // w
        0x44, 0x28, 0x10, 0x28, 0x44,   // x
        0x1C, 0xA0, 0xA0, 0xA0, 0x7C,   // y
        0x44, 0x64, 0x54, 0x4C, 0x44,   // z

        0x00, 0x10, 0x7C, 0x82, 0x00,   // {
        0x00, 0x00, 0xFF, 0x00, 0x00,   // |
        0x00, 0x82, 0x7C, 0x10, 0x00,   // }
        0x00, 0x06, 0x09, 0x09, 0x06    // ~ (Degrees)
};



/**************************************************************************************************
                                void OLED_Init()
 ***************************************************************************************************
 * I/P Arguments:  none
 * Return value : none

 * description  :This function is used to initialize the OLED in the normal mode.
                After initializing the OLED, It clears the OLED and sets the cursor to first line first position. .

 **************************************************************************************************/
void OLED_Init(void)
{ 
  oledSendCommand(0xa8);
  oledSendCommand(0x3f);
  oledSendCommand(0xd3);
  oledSendCommand(0x00);
  oledSendCommand(0x40);
	oledSendCommand(0xa1);
	oledSendCommand(0xc8);
	oledSendCommand(0xda);
	oledSendCommand(0x12);
	oledSendCommand(0x81);
	oledSendCommand(0x7f);
	oledSendCommand(0xa4);
	oledSendCommand(0xa6);
	oledSendCommand(0xd5);
	oledSendCommand(0x80);
	oledSendCommand(0x8d);
	oledSendCommand(0x14);
	oledSendCommand(0xaf);
}




/***************************************************************************************************
                       void OLED_DisplayChar( char ch)
 ****************************************************************************************************
 * I/P Arguments: ASCII value of the char to be displayed.
 * Return value    : none

 * description  : This function sends a character to be displayed on LCD.
                  Any valid ascii value can be passed to display respective character

 ****************************************************************************************************/
void OLED_DisplayChar(int8_t ch)
{
    uint8_t i=0;
    int index;

    if(ch!='\n') {  /* TODO */ 
        index = ch;
        index = index - 0x20;
        index = index * FONT_SIZE; // As the lookup table starts from Space(0x20)

        for(i = 0; i < FONT_SIZE; i ++)
            oledSendData(OledFontTable[index + i]); /* Get the data to be displayed for LookUptable*/

        oledSendData(0x00); /* Display the data and keep track of cursor */
    }
}

/***************************************************************************************************
                       void OLED_DisplayString(char *ptr_stringPointer_u8)
 ****************************************************************************************************
 * I/P Arguments: String(Address of the string) to be displayed.
 * Return value    : none

 * description  :
               This function is used to display the ASCII string on the lcd.
                 1.The ptr_stringPointer_u8 points to the first char of the string
                    and traverses till the end(NULL CHAR)and displays a char each time.

 ****************************************************************************************************/
void OLED_DisplayString(uint8_t *ptr)
{
    while(*ptr)
        OLED_DisplayChar(*ptr++);
}

/***************************************************************************************************
                void OLED_SetCursor(char v_lineNumber_u8,char v_charNumber_u8)
 ****************************************************************************************************
 * I/P Arguments: char row,char col
                 row -> line number(line1=1, line2=2),
                        For 2line LCD the I/P argument should be either 1 or 2.
                 col -> char number.
                        For 16-char LCD the I/P argument should be between 0-15.
 * Return value    : none

 * description  :This function moves the Cursor to specified position

                   Note:If the Input(Line/Char number) are out of range
                        then no action will be taken
 ****************************************************************************************************/
void OLED_SetCursor(uint8_t page, uint8_t cursorPosition)
{
  cursorPosition = cursorPosition;
  oledSendCommand(0x0f&cursorPosition);
	oledSendCommand(0x10|(cursorPosition>>4));
	oledSendCommand(0xb0|page);
}

/***************************************************************************************************
                         void OLED_Clear(void)
 ****************************************************************************************************
 * I/P Arguments: none.
 * Return value    : none

 * description  :This function clears the LCD and moves the cursor to beginning of first line
 ****************************************************************************************************/
void OLED_Clear(void)
{	
    uint8_t oled_clean_col , oled_clean_page;
	for(oled_clean_page = 0 ; oled_clean_page < 8 ; oled_clean_page++) {
        OLED_SetCursor(oled_clean_page,0);
		for(oled_clean_col= 0 ; oled_clean_col < 128 ; oled_clean_col ++) {
            oledSendData(0);
		}
	}
}



/********************************************************************************
                Local FUnctions for sending the command/data
 ********************************************************************************/
void oledSendCommand(uint8_t cmd)
{
	uint8_t data_write[2];
	uint8_t data_read[2];
	data_write[0] = SSD1306_COMMAND;
	data_write[1] = cmd;

    hx_drv_i2cm_write_data(USE_SS_IIC_X, SSD1306_ADDRESS, &data_write[0], 1, &data_write[1], 1); 
}

int32_t oledSendData(uint8_t cmd)
{

	uint8_t data_write[2];
	uint8_t data_read[2];
	data_write[0] = SSD1306_DATA_CONTINUE;
	data_write[1] = cmd;

    int32_t test = hx_drv_i2cm_write_data(USE_SS_IIC_X, SSD1306_ADDRESS, &data_write[0], 0, &data_write[0], 2); 
    return test;
}
/********************************************************************************
                homemade function
 ********************************************************************************/
#define HeartNUM 22
uint8_t HeartLocation[22][2] = {
    {0, 47}, {0, 52}, {0, 72}, {0, 77}, {1, 42}, 
    {1, 57}, {1, 67}, {1, 82}, {2, 37}, {2, 62}, 
    {2, 87}, {3, 37}, {3, 87}, {4, 42}, {4, 82},
    {5, 47}, {5, 77}, {6, 52}, {6, 72}, {7, 57},
    {7, 62}, {7, 67}};

uint8_t HeartData = 0xff;

void DisplayHeart(void){
    for(int j = 0; j < HeartNUM; j++){
        OLED_SetCursor(HeartLocation[j][0], HeartLocation[j][1]);
        for(int i = 0; i < FONT_SIZE; i++)
            oledSendData(HeartData);
        oledSendData(0x00);
    }
}
#define TickNUM 5

uint8_t TickLocation[TickNUM][2] = {
    {4, 2}, {5, 7}, {4, 12}, {3, 17}, {2, 22}
};

uint8_t TickData = 0xff;


void DisplayTick(void){
    for(int j = 0; j < TickNUM; j++){
        OLED_SetCursor(TickLocation[j][0], TickLocation[j][1]);
        for(int i = 0; i < FONT_SIZE; i++)
            oledSendData(TickData);
        oledSendData(0x00);
    }
}

#define MarNUM 9

uint8_t MarkLocation[MarNUM][2] ={
    {2, 102}, {2, 122}, {3, 107}, {3, 117}, {4, 112},
    {5, 107}, {5, 117}, {6, 102}, {6, 122}
    };

uint8_t MarkData = 0xff;

void DisplayExclamationMark(void){
    for(int j = 0; j < MarNUM; j++){
        OLED_SetCursor(MarkLocation[j][0], MarkLocation[j][1]);
        for(int i = 0; i < FONT_SIZE; i++)
            oledSendData(MarkData);
        oledSendData(0x00);
    }
}

void DisplayResult(char result){
    OLED_Clear();
    switch (result)
    {
    case 'N':{
        DisplayHeart();
        DisplayTick();
        break;
    }

    case 'Y':{
        DisplayHeart();
        DisplayExclamationMark();
        break;
    }

    case 'R':{
        DisplayReady();
        break;
    }

    default:{
        DisplayHeart();
        break;
    }
    }
}
#define ReadyNum 63
uint8_t ReadyLocation[ReadyNum][2] ={
    {0, 12}, {0, 17}, {0, 22}, {3, 12}, {3, 17}, {3, 22}, {6, 12}, {6, 17}, {6, 22},
    {1, 12}, {2, 12}, {4, 22}, {5, 22},  //13
    {0, 37}, {1, 37}, {2, 37}, {3, 37}, {4, 37}, {5, 37}, {6, 37},
    {6, 42}, {6, 47}, // 9
    {0, 62}, {1, 62}, {2, 62}, {3, 62}, {4, 62}, {5, 62}, {6, 62},
    {0, 87}, {1, 87}, {2, 87}, {3, 87}, {4, 87}, {5, 87}, {6, 87},
    {0, 67}, {0, 72}, {3, 67}, {3, 72}, {6, 67}, {6, 72}, 
    {0, 92}, {0, 97}, {3, 92}, {3, 97}, {6, 92}, {6, 97}, //26
    {0, 112}, {1, 112}, {2, 112}, {3, 112}, {4, 112}, {5, 112}, {6, 112}, 
    {0, 117}, {0, 122}, {0, 127}, {1, 127}, {2, 127}, {3, 117}, {3, 122}, {3, 127}, //15
    };

uint8_t ReadyData = 0xff;

void DisplayReady(void){
    for(int j = 0; j < ReadyNum; j++){
        OLED_SetCursor(ReadyLocation[j][0], ReadyLocation[j][1] - 10);
        for(int i = 0; i < FONT_SIZE; i++)
            oledSendData(ReadyData);
        oledSendData(0x00);
    }
}

#define AHINum 44
uint8_t AHILocation[AHINum][2] = {
    {0, 17}, {0, 22},
    {1, 12}, {2, 12}, {3, 12}, {5, 12}, {6, 12},
    {1, 27}, {2, 27}, {3, 27}, {5, 27}, {6, 27},
    {4, 12}, {3, 17}, {3, 22}, {4, 27}, //16
    {0, 37}, {1, 37}, {2, 37}, {3, 37}, {4, 37}, {5, 37}, {6, 37},
    {0, 47}, {1, 47}, {2, 47}, {3, 47}, {4, 47}, {5, 47}, {6, 47},  
    {3, 42},///15
    {0, 62}, {1, 62}, {2, 62}, {3, 62}, {4, 62}, {5, 62}, {6, 62},
    {6, 57}, {6, 67},
    {0, 57}, {0, 67},///11
    {2, 77}, {4, 77}//2
};
uint8_t AHIData = 0xff;

uint8_t NumLocation[15][2] = {
   {0, 5}, {1, 5}, {2, 5}, {3, 5}, {4, 5},
   {0, 10}, {1, 10}, {2, 10}, {3, 10}, {4, 10},
   {0, 15}, {1, 15}, {2, 15}, {3, 15}, {4, 15},
};

uint8_t NumData[10][15] ={
    {255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255,},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255,},
    {255, 0, 255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 0, 255,},
    {255, 0, 255, 0, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255,},
    {255, 255, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 255, 255, 255,},
    {255, 255, 255, 0, 255, 255, 0, 255, 0, 255, 255, 0, 255, 255, 255,},
    {255, 255, 255, 255, 255, 255, 0, 255, 0, 255, 255, 0, 255, 255, 255,},
    {255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255, 255,},
    {255, 255, 255, 255, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255,},
    {255, 255, 255, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255,},
};

void DisplayAHI(int num){
    OLED_Clear();
    for(int j = 0; j < AHINum; j++){
        OLED_SetCursor(AHILocation[j][0], AHILocation[j][1] - 10);
        for(int i = 0; i < FONT_SIZE; i++)
            oledSendData(AHIData);
        oledSendData(0x00);
    }
    for(int k = 0; k < 2 ; k++){
        for(int j = 0; j < 15; j++){
        OLED_SetCursor(NumLocation[j][0] + 1, 77 + NumLocation[j][1] + k * 20);
        for(int i = 0; i < FONT_SIZE; i++)
            if(k == 0)
                oledSendData(NumData[(num / 10)][j]);
            else
                oledSendData(NumData[(num % 10)][j]);
        oledSendData(0x00);
    }
    }
   
    if(num >= 5 && num < 15){
        OLED_SetCursor(7, 46);
        OLED_DisplayString("Mild OSA");
    }
    else if(num >= 15 && num < 30){
        OLED_SetCursor(7, 20);
        OLED_DisplayString("Moderate OSA");
    }
    else if(num >= 30){
        OLED_SetCursor(7, 40);
        OLED_DisplayString("Severe OSA");
    }
    else{
        OLED_SetCursor(7, 54);
        OLED_DisplayString("NO OSA");
    }

}
```
- Initial MAX86150 to send ECG data.


```c
void InitMax86150(void){
    InitI2C();
    uint8_t initial_data_write[8][2] = { 
                                         {0x02, 0x80}, //Interrupt Enable 1
                                         {0x03, 0x04}, //Interrupt Enable 2
                                         {0x08, 0x10}, //FIFO Configuration 
                                         {0x09, 0x09}, //FIFO Data Control Register 1
                                         {0x0a, 0x00}, //FIFO Data Control Register 2
                                         {0x3c, 0x03}, //ECG Configuration 1   //0x02   0x03
                                         {0x3e, 0x00}, //ECG Configuration 3    //0x0D  0x00
                                         {0x0d, 0x04}}; //System Control; 
    for(int i = 0; i < 8; i++){
        Max86150_WriteData(initial_data_write[i][0], initial_data_write[i][1]);
    }
    board_delay_ms(500);
        OLED_Init();

	OLED_Clear();                        
	OLED_SetCursor(0, 0);
    OLED_DisplayString("Welcome To ARC EVK");

	OLED_SetCursor(1, 0);
	for(oled_i = 0; oled_i < 128; oled_i ++)
		oledSendData(oled_i);
}
```
- Initial OLED1306 to show result.
```c
OLED_Init();

OLED_Clear();                        
OLED_SetCursor(0, 0);
OLED_DisplayString("Welcome To ARC EVK");

OLED_SetCursor(1, 0);
for(oled_i = 0; oled_i < 128; oled_i ++)
	oledSendData(oled_i);
```

- Initial GMA303KU to send 3-axis acceleration.
```c
int16_t accel_x;
int16_t accel_y;
int16_t accel_z;
int16_t accel_t;

uint8_t xg_sign;
uint8_t yg_sign;
uint8_t zg_sign;
uint8_t temp_sign;

int16_t xg_10x;
int16_t yg_10x;
int16_t zg_10x;
int16_t temp_10x;
int main(){

    //your code
     iic1_ptr = hx_drv_i2cm_get_dev(USE_SS_IIC_0);
    iic1_ptr->iic_open(DEV_MASTER_MODE, IIC_SPEED_STANDARD); 
    uint8_t chip_id = GMA303KU_Init();
    board_delay_ms(100);

    if(chip_id == 0xA3)
        sprintf(uart_buf, "Chip ID: 0x%2X | OK\r\n\n", chip_id);    //It should be 0xA3
    else 
        sprintf(uart_buf, "Chip ID: 0x%2X | Error\r\n\n", chip_id);    //It should be 0xA3
    uart0_ptr->uart_write(uart_buf, strlen(uart_buf));
    board_delay_ms(10);
    //your code
}
```
#### Project programming
-Open tflitemicro_algo.cpp, put layers we have in model.
```cpp
	static tflite::MicroMutableOpResolver<8> micro_op_resolver;
	micro_op_resolver.AddMul();
	micro_op_resolver.AddAdd();
	micro_op_resolver.AddConv2D();
	micro_op_resolver.AddFullyConnected();
	micro_op_resolver.AddReshape();
	micro_op_resolver.AddMaxPool2D();
	micro_op_resolver.AddSoftmax();
	micro_op_resolver.AddRelu();
```
-Open model_settings.cpp, and change label.
```cpp
const char kCategoryLabels[kCategoryCount] = { 'N', 'Y', };
```

```cpp
```
```cpp
```
```cpp
```
### Python work for deep learning
#### Loading dataset and pre-processing
- Includes module
```python
import wfdb
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization,Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
```

- Load testing dataset

```python
adr =1
if adr ==1:
 change_link='/content/drive/MyDrive/CNN/ECG_model/sleep_apena_detect/apnea-ecg-database-1.0.0/'
elif adr == 2:
  change_link='/content/drive/MyDrive/ecg_data/apnea-ecg-database-1.0.0/'

x_test=[]
y_test=[]

test_string=['x01','x02','x03','x04','x05','x06','x07','x08','x09','x10','x11','x12',
             'x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24',
             'x25','x26','x27','x28','x29','x30','x31','x32','x33','x34','x35']

             
test_label_count=[522,468,464,481,504,449,508,516,507,509,456,526,505,489,497,514,399,458,486,
                  512,509,481,526,428,509,519,497,494,469,510,556,537,472,474,482]

test_label=np.loadtxt(change_link+'test-dataset-annos.txt',delimiter='\t',dtype=np.str)



tmp=[]

for i in test_label:
    tmp+=list(i)


for i in range(len(tmp)):
    if tmp[i]=='N':
        tmp[i]=0
    else:
        tmp[i]=1
y_test=np.array(tmp,dtype=np.int8)

count=0
for k in test_string:
    record = wfdb.rdrecord(record_name=change_link+k,return_res=16,physical=True)
    ecg_signal = record.p_signal
    ecg_signal=np.delete(ecg_signal,np.s_[((test_label_count[count])*6000):])
    x_test=np.append(x_test,ecg_signal)
    x_test = x_test.astype(np.float32)
    count+=1
```
- Load training dataset
```python
x_train=[]
y_train=[]
list_string=['a01','a02','a03','a04','a05','a06','a07','a08','a09','a10','a11','a12',
             'a13','a14','a15','a16','a17','a18','a19','a20','b01','b02','b03','b04',
             'b05','c01','c02','c03','c04','c05','c06','c07','c08','c09','c10']

for k in list_string:

    record = wfdb.rdrecord(record_name=change_link+k,return_res=16,physical=True)
    ann = wfdb.rdann(change_link+k,'apn')

    ecg_signal_label=ann.symbol
    ecg_signal_label=np.array(ecg_signal_label)

    ecg_signal = record.p_signal


    ecg_signal=np.delete(ecg_signal,np.s_[((len(ecg_signal_label)-1)*6000):])
    x_train=np.append(x_train,ecg_signal)
    x_train = x_train.astype(np.float32)

    for i in range(len(ecg_signal_label)):
        if ecg_signal_label[i]=='N':
            ecg_signal_label[i]=0
        else:
            ecg_signal_label[i]=1
    ecg_signal_label=np.array(ecg_signal_label,dtype=np.int8)
    ecg_signal_label=np.delete(ecg_signal_label,np.s_[(len(ecg_signal_label)-1):])
    y_train=np.append(y_train,ecg_signal_label)
```
- Merge two dataset in one
```python
tmp_data=np.append(x_train,x_test,)
tmp_label=np.append(y_train,y_test)
```
- Dataset normalization
```python
tmp_data_max_abs=np.max(np.abs(tmp_data))
print(tmp_data)
tmp_data=tmp_data/tmp_data_max_abs
print(tmp_data)
```
- Split dataset
```python
tmp_data=np.reshape(tmp_data,(int(tmp_data.size/6000),6000,1,1))

x_train, x_test, y_train, y_test = train_test_split(tmp_data, tmp_label, test_size=0.1)
x_train.shape,y_train.shape ,x_test.shape , y_test.shape
```
- Flatten Label
```python
y_train = y_train.flatten()
y_test = y_test.flatten()
```
- Encoder Label
```python
num_classes=2
y_train_encoder = sklearn.preprocessing.LabelEncoder()
y_train_num = y_train_encoder.fit_transform(y_train)
y_train_wide = tensorflow.keras.utils.to_categorical(y_train_num, num_classes)
y_test_num = y_train_encoder.fit_transform(y_test)
y_test_wide = tensorflow.keras.utils.to_categorical(y_test_num, num_classes)
```
#### Setup model
- Use 2D CNN to deal with this problem
```python
model_ecg = Sequential()

model_ecg.add(BatchNormalization(input_shape=(6000,1,1)))

model_ecg.add(Conv2D(8,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(Conv2D(8,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(MaxPooling2D(pool_size=(2,1),padding="same"))

model_ecg.add(Conv2D(16,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(Conv2D(16,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(MaxPooling2D(pool_size=(2,1),padding="same"))

model_ecg.add(Conv2D(32,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(Conv2D(32,kernel_size=(2,1),padding="same", activation='relu',strides=2,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model_ecg.add(MaxPooling2D(pool_size=(2,1),padding="same"))

model_ecg.add(Dropout(0.5))

model_ecg.add(BatchNormalization())
model_ecg.add(Flatten())
model_ecg.add(Dropout(0.5))

model_ecg.add(Dense(64,activation='relu'))
model_ecg.add(Dense(2,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),activation='softmax'))

model_ecg.summary()
```
- Choose optimizer and loss function
```python
opt = tensorflow.keras.optimizers.Adam(lr=0.001)
model_ecg.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```
- Select model's hyper parameter
```python
batch_size = 16
epochs = 300
```

- Get ready to train model
```python
best_weights_filepath_ecg = './best_weights_ecg_32layer.hdf5'
mcp_ecg = ModelCheckpoint(best_weights_filepath_ecg, monitor="val_accuracy",
                      save_best_only=True, save_weights_only=False)
history = model_ecg.fit(x_train, 
                        y_train_wide,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[mcp_ecg])
```
- Plot accuracy and loss
```python
loss = history.history['loss']
val_loss = history.history['val_loss']

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

plt.figure(figsize=(6,12))

plt.subplot(2,1,1)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,epochs)[0::100])
plt.title('Training and Validation Loss vs Epochs')
plt.legend()

plt.subplot(2,1,2)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(acc, 'blue', label='Training Accuracy')
plt.plot(val_acc, 'green', label='Validation Accuracy')
plt.xticks(range(0,epochs)[0::100])
plt.title('Training and Validation Accuracy vs Epochs')
plt.legend()
plt.savefig("plots_perf.svg")
plt.show()
```
- Check our confusion matrix 
```python
model_ecg.load_weights('best_weights_ecg_32layer.hdf5')
y_pred = model_ecg.predict(x_test)
predict_test=np.argmax(y_pred, axis=1)
predict_test=predict_test.reshape(predict_test.shape[0],1)
cm=confusion_matrix(y_test_num, predict_test)
cm
```
- Check test data accuracy
```python
(cm[1,1]+cm[0,0])/(cm[1,1]+cm[1,0]+cm[0,0]+cm[0,1])
```
- Save module and weight
```python
model_ecg.save_weights('model_weights.h5')
model_ecg.save('model_weights.h5')
```
- Convert 'h5' file to 'tflite' file
```python
import tensorflow as tf
model = tf.keras.models.load_model('model_weights.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model_ecg)
tflite_model = converter.convert()
converter.inference_input_type, converter.inference_output_type
import pathlib
generated_dir = pathlib.Path("generated/")
generated_dir.mkdir(exist_ok=True, parents=True)
converted_model_file = generated_dir/"ecg_model.tflite"
converted_model_file.write_bytes(tflite_model)  
```
- Check 'tflite' file's accuracy
```python
import tensorflow as tf
max_samples = 17233
converted_model_file="generated/ecg_model.tflite"
interpreter = tf.lite.Interpreter(model_path=str(converted_model_file))
interpreter.allocate_tensors()

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    scale, zero_point = interpreter.get_output_details()[0]['quantization']

    prediction_values = []
    
    for test_image in x_test[:max_samples]:
        # Pre-processing: add batch dimension, quantize and convert inputs to int8 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0) #.astype(np.float32)
        test_image = np.float32(test_image)
        interpreter.set_tensor(input_index, test_image)

        interpreter.invoke()

        # Find the letter with highest probability
        output = interpreter.tensor(output_index)
        result = np.argmax(output()[0])
        prediction_values.append(result)
    
    accurate_count = 0
    for index in range(len(prediction_values)):
        if prediction_values[index] == y_test[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_values)
    return accuracy * 100

print(str(evaluate_model(interpreter)) + "%")
```
- Open **Cygwin64 Terminal** press command
```c
xxd -i model.tflite > model.h
```
And you get .h file to put in ARC EM9D
