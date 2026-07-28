on run argv
	if (count of argv) is not 2 then error "usage: export_pptx_to_pdf.applescript INPUT.pptx OUTPUT.pdf"
	set sourceFile to POSIX file (item 1 of argv)
	set outputFile to POSIX file (item 2 of argv)

	tell application "Microsoft PowerPoint"
		open sourceFile
		delay 1
		set sourceDeck to active presentation
		save sourceDeck in outputFile as save as PDF
		close sourceDeck saving no
	end tell
end run
