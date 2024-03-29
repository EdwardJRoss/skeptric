---
categories:
- excel
date: '2020-06-04T18:08:02+10:00'
image: /images/mail_merge.png
title: Mail merge to PDF Files
---

A friend needed to generate a hundred contracts and their HR information system wasn't working properly.
I helped them implement a workaround solution by using mail merge to generate a PDF for every contract, which saved them a lot of time filling in the details of each contract.
I couldn't automatically generate the PDF despite some efforts, but using mail merge was much quicker and more reliable than filling in all the contract details manually into the template.
She used the Preview to scroll through each contract manually and "Saved As" a separate PDF which took about 40 minutes.

Microsoft Office offers mail merge (under the Mailing ribbon) which lets you generate documents for printing or email that fill in individual details from an Excel spreadsheet (or other datasource).
Unfortunately there's no way to generate separate Word or PDF files directly from mail merge.
I found a [macro](https://www.msofficeforums.com/mail-merge/21803-mailmerge-tips-tricks.html) to do it.
When I first ran it raised an error and I had to remove the line `If Err.Num = 5631 Then Err.Clear`.
Then it worked perfectly, taking fields `First_Name` and `Last_Name` from the spreadsheet and producing a PDF file `<Last_Name>_<First_Name>.pdf` (and a corresponding Word file).

```
Sub Merge_To_Individual_Files()
' Sourced from: https://www.msofficeforums.com/mail-merge/21803-mailmerge-tips-tricks.html
Application.ScreenUpdating = False
Dim StrFolder As String, StrName As String, MainDoc As Document, i As Long, j As Long
Const StrNoChr As String = """*./\:?|"
Set MainDoc = ActiveDocument
With MainDoc
  StrFolder = .Path & "\"
  With .MailMerge
    .Destination = wdSendToNewDocument
    .SuppressBlankLines = True
    On Error Resume Next
    For i = 1 To .DataSource.RecordCount
      With .DataSource
        .FirstRecord = i
        .LastRecord = i
        .ActiveRecord = i
        If Trim(.DataFields("Last_Name")) = "" Then Exit For
        'StrFolder = .DataFields("Folder") & "\"
        StrName = .DataFields("Last_Name") & "_" & .DataFields("First_Name")
      End With
      On Error GoTo NextRecord
      .Execute Pause:=False
      For j = 1 To Len(StrNoChr)
        StrName = Replace(StrName, Mid(StrNoChr, j, 1), "_")
      Next
      StrName = Trim(StrName)
      With ActiveDocument
        'Add the name to the footer
        '.Sections(1).Footers(wdHeaderFooterPrimary).Range.InsertBefore StrName
        .SaveAs FileName:=StrFolder & StrName & ".docx", FileFormat:=wdFormatXMLDocument, AddToRecentFiles:=False
        ' and/or:
        .SaveAs FileName:=StrFolder & StrName & ".pdf", FileFormat:=wdFormatPDF, AddToRecentFiles:=False
        .Close SaveChanges:=False
      End With
NextRecord:
    Next i
  End With
End With
Application.ScreenUpdating = True
End Sub
```

Unfortunately my friend was on Mac and this script didn't work.
I stepped into the debugger and noticed the `.recordcount` was `-1`.
Reading the [docs](https://docs.microsoft.com/en-us/office/vba/api/word.mailmergedatasource.recordcount) it seems this happens when it can't detemine the number of records, which was confirmed on [stackoverflow](https://stackoverflow.com/questions/37921973/word-mail-merge-recordcount-returns-1).

I tried manually overriding the recordcount to 5, but then it turned out the `.DataFields` was empty.
Something weird was happening on Mac Office and we were reaching the point of diminishing returns to fix it.

Instead we worked out if we saved the PDF from the preview it came out correctly.
It was relatively quick to repeatedly jump to next record in preview and then Save as a PDF file, copying the name from the preview to use as the filename.
This took about 40 minutes for 100 contracts, which is faster than I could have debugged the macro.
