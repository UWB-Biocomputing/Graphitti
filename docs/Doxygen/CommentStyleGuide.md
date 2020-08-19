## Doxyfile Settings
change from default: 
```
JAVADOC_AUTOBRIEF      = NO
```
to: 
```
JAVADOC_AUTOBRIEF      = YES
```

If the `JAVADOC_AUTOBRIEF` tag is set to `YES` then doxygen will interpret the
first line (until the first dot) of a Javadoc-style comment as the brief
description. If set to `NO`, the Javadoc-style will behave just like regular Qt-
style comments (thus requiring an explicit `@brief` command for a brief
description.)

***********************************

## BG Commenting 
#### Class Descriptions
```
/**
 * @file TheClass.cpp
 * 
 * @ingroup DirectoryName
 *
 * @brief Write a short description of the class that orients the reader
 *
 * Use this space for more details. 
 * If the class needs multiple paragraphs, this is ok. 
 * 
 * Here is the second paragraph.
 * Do not exceed 80 chars per line
 */
```

#### Function Comments
BG uses triple-slash style commenting for functions. 
This is why `JAVADOC_AUTOBRIEF` needs to be set to `YES`

```
/// Brief description which ends at this dot. Details follow
/// here.
```

#### In-line Comments
If a comment follows the code it is referring to, include a `<` after 
the triple-slash as follows: 

```
int var; ///< Brief description after the member
```

#### Method Section Dividers
If there is a section of methods/functions, 
#### ***todo: finish this ***
	/** @name Math Operations
     
	 Math operations. For efficiency's sake, these methods will be
	 implemented as being "aware" of each other (i.e., using "friend"
	 and including the other subclasses' headers).
	 */
	//@{

***********************************

## BG Commenting not for Doxygen Purposes
#### ToDo's and Errors 
Commenting ToDo's can be done outside of the Doxygen format. 
``` 
// ToDo: For example, this will not generate Doxygen documentation
```