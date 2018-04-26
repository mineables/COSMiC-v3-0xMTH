COSMIC [the Community Open-Source Miner, Including CUDA] for PoW Tokens (0xBitcoin)
Version 2.10-pre5 + 'HashBurner' Build by LtTofu a.k.a. Mag517, Version 3.4t (release candidate)
------------------------------------------------------------------------------------------------

Please consider a donation to help support my development! Love you guys! :)
ETH/0xBTC/Other Tokens: 0x7C4AADEC857E13E4a5642B4041a2f36274fFE8ce

** Important Note! **
	Default intensity set to 24 for lower-end hardware: configure a custom intensity by
	following the instructions later in this Readme to maximize performance. Too high an
	intensity selection might cause diminishing returns in performance, lower performance,
	kernel launch failures, crashes or other undesired behavior. Too LOW intensity might
	underutilize your GPU. If you experience slowdown in your GUI (e.g. Windows), lowering
	this might help alleviate it.

Changes added in COSMiC V3.4t (this version):
---
- (Bugfix) Fixed `account` commands to list/select/create accounts, which broke sometime around 2.10.0pre5.
	   Be sure to keep your miner up to date! :)
- (General) Now notifies users of new challenges received by the miner. Other UI tweaks.
- (General) Auto-donate a.k.a. devfee can now be configured in multiples of 0.5%.

Changes added in COSMiC V3.x (previous versions of COSMiC-V3.x series):
---

- (Speedup) Started implementing a shortcut which flips the high and low bytes of a state[] element to get
		the same result as rotating it by a magnitude of 32 in either direction, then uses funnel
		shift to get the rest of the way to the 'destination'. ex. Reduces ROTL by 28 to just 4
- (General) WIP: Optimizations are chosen based on the detected card, set in h_HBoptimizations[] and sent to
		device where they can be implemented. Display optimizations in use. Some currently hard-coded
		to provide the best balance of performance on Maxwell Gen2/Pascal.
- (General) Added configurable Automatic Donation (a.k.a. 'DevFee') with default of 2%/1% depending on the
		archive(7zip) you choose, turn it off, or set a higher one if you really like this build :)
		Donations support me further developing COSMiC and will be shared with contributors to the
		code it's based upon! Read on for more details re: configuring donation.
- (Speedup) Do 32-bit XOR (in PTX/ASM) instead of 64-bit for Round Constants that are half leading zeroes
			(partially implemented: may be expanded in next release after performance testing)
- (Speedup) Quicker keccak() hashing function init- more use of multiplication on nonce instead of ROTL64
- (Speedup) keccak() hashing function's helper functions tuned for better performance.
- (Speedup) Round Constants (RC) LUT defined in keccak() hash function, not constant memory
- (Speedup) RC 0 and RC 23 contents are now immediates, no lookup needed.
- (Speedup) More use of permutation in the place of rotations.
- (Speedup) Testing ROTL64-replacement for speed in keccak() hashing function's init/nonce calcs
- (General) Solutions are now displayed, including invalids (which are not submitted).
- (General) Devfee of ~2% to support 0xBTC miner devs, adjustable (see later in this document).
- (General) Should produce less invalid shares (as interpreted at the pool level.)
- (General) Nonce rotation lookup table removed for performance testing.
- (General) Small UI changes.

Changes added in COSMiC V2 (previous version):
---

- (Speedup) Operations combined with LOP3 instruction in new PTX/ASM-based function.
- (Speedup) Rotate operation results on the Nonce are precomputed and stored in a lookup table
	    so they only need to be performed once per kernel regardless of how many times it has
	    to run before a result is found. LUT is passed to keccak() as function argument.
	    Working on a faster version.
- (Speedup) Faster ASM/PTX variable-magnitude rotation functions (left and right).
- (Speedup) Slightly tighter/faster single-position left rotation.
- (Speedup) Implemented permutations (PRMT) where possible to reduce instructions performed
	    in the frequent rotation operations. Still some places I can improve, though!
- (Speedup) Faster bswap_64() function written in PTX/ASM.
- (Speedup/General) No more spilled registers from "chi" and other functions with SM 6.x.
		    Reimplemented LOP3 instructions.
- (General) REMOVED for testing in v2.1t : reset of `cnt` every solution found/new challenge
- (General) Default intensity set to 24 (was 27 in HB enhancements V1). Change it in ./0xbtc.conf,
	    described in this file. Note that this file might be deprecated soonish.
- (To Do) Some spilled registers with SM 5.2 code- does not appear to affect performance
	  on SM 6.x (Pascal) device tested- please report your results!
- (To Do) More graceful handling of cases where multiple CUDA-capable GPUs are installed.
- (To Do) Network code improvements for better pool compatibility.
- (To Do) More rotation trickery for speed.

.. and quite a few more little tweaks I might've forgotten to list.


Changes added by COSMiC V1 (previous version):
---

- (Speedup) Bidirectional Rotation: State values rotated right rather than left resulting in a rotation
	  by a magnitude of 32 or less. 
	  Ex. rotate right by 1 instead of left by 63, right by 20 instead of left by 44.
	  This also means a simpler, shorter version of ROTL64/ROTR64 can be used.
- (Speedup) New minimal ROTL64/ROTR64 functions in ASM/PTX utilizing Funnel Shifter instead of Shifts/OR.
	  (Seems to be a little faster than the previous idiom/default compiler optimization on Pascal arch.)
- (Speedup) Midstate filled out in gpu_mine kernel and passed to keccak() as function argument to avoid the
	  100+ cycle penalty in accessing Constant memory (see: intMid[25])
- (Speedup) Applied threading approach from pre-SM 5.2 control path for SM 5.2+ devices (Azlehria)
- (Speedup) Slightly faster device initialization w/ bidirectional rotation, other tweaks
- (Speedup) Used immediates to reduce reads from RC[25] in Constant memory by ~8.33%
- (Fix) Stabilized hashrate somewhat (now 0 spilled registers in SM 6.x)
- (Fix) Rolled back LOP3 chi() function due to register spillage - replacement forthcoming
- (Fix) Private keys should no longer be stored in plaintext in Vault file (Azlehria)
- (General) Somewhat different randomization approach in initialization
- (General) User interface changes including device identifier, available CUDA version (GPU+driver)

To use:
---
- Run Cosmic-0xBTC.exe from a command prompt or ./Cosmic-0xBTC.exe from a PowerShell.
- Execute these commands (without `)
	account select 0x??????????????? or account new
	(if creating a new account, do ` accounts list ` and record your private key.
	
	pool select http://hostname:port
	(suggestions: http://mike.rs:8586  0xpool.io  http://tokenminingpool.com:8586)

	` pool mine `   (CPU only) or   ` pool mine cuda `   (CUDA devices)

- For more options, type    ` help ` at the command: prompt.


To use the unpackaged version (may be faster on your machine, includes dev environment):
---
- Run  ` node index.js ` at command prompt or PowerShell.
- Same commands above for pool and account settings apply. 

Customizing your Intensity:
---
- Format for the 0xbtc.conf file (included) is:           intensity cuda-device
- See the included file. If not present, a hardcoded intensity will be used. Normally defaults
	to CUDA device 0.
- Suggested intensity for a GTX 1060: 26-27. Higher intensities update the hashrate meter more slowly.
	Lower intensities update it more quickly, which might be helpful if you are altering the source.

Please report your ideal intensity and device type so that I can implement automatic detection/intensity choice.

Building from Source (on Windows):
- The source code and node modules you'll need are included. You will have to install npm and, possibly:
	- Install Git and Python for Windows

	` npm install windows-build-tools `	
	... from the directory where you extracted this archive, at a command prompt/PowerShell.

- Do ` node-gyp clean
       node-gyp configure
       node-gyp build     `
- Or ` npm install  `
- Or ` ./buildandrun  ` or `  ./buildandrun.bat   `

I will be updating this section with more thorough info on how to build in Windows if you hit a snag.
You can also get some tips from this thread: https://www.reddit.com/r/0xbitcoin/comments/8381br/guideongoing_how_to_build_the_opensource_cuda/

Adjusting the Automatic Donation a.k.a. DevFee:
---
The DevFee helps support the developers of 0xBTC miners, including this one and will be distributed to all
contributors to the published/released code. About 2% of shares, a little less really will by default go to pay
the DevFee. Unlike most miners, the mining process does not stop and you remain connected to your chosen pool at
all times! This ensures there are no hiccups and your mining speed is NOT affected. :)

If you'd like to change the % of the Donation a.k.a. DevFee or disable it entirely (please think of the poor devs!):
- Extract the source code from the BitBucket ( https://bitbucket.org/LieutenantTofu/cosmic-v3/ ) or archive (7z)
	into a convenient location, which I'll call the 'root' below.
- Open lib/0xbitcoinminer-accel.js. At the top, look for ` const donationPercent = 2; ` which means 2% Donation a.k.a. DevFee.
- Change this number to pick a higher or lower percentage. At 2, 2% of shares, occurring semirandomly, will be donated.
- You can also change the DevFee address in the submitNewMinedBlock() function, around line 157, to donate to a specific
	developer, divide up your earnings across multiple ETH accounts or mine on behalf of a friend.
- Build using the above instructions or simply run with the `  node index.js  ` command from your terminal.
- Future builds will make this configurable with a command inside the program. I'm still learning NodeJS. :)

Troubleshooting:
---

Q: "How do I mine with multiple GPUs?"
A: Adding this info right away! In the meantime, it's a hot topic of conversation in #support on the Discord.
There are a few ways to do this right now- I'll also be supplying a sample batch file to get you started along with
some of the other approaches. Thanks for your patience!

Q: "The window closes immediately"
A: You need to run from a command prompt to see what the error was. Type 'cmd' at the Run prompt (Windows Key + R) or
	pick PowerShell from the File menu in the location where you extracted this archive.

Q: "The console is displaying what looks like garbage."
A: This is a known issue in Windows 7's command prompt. Try installing a different shell or use a newer OS on the machine
	where you mine. There's a port of Bash to Windows- you might give that a shot.

As any issues are reported, this section will be updated with the next build.


Credits:
---
Donations encouraged to help support further development of the 0xBitcoin project and this miner!
Find the 0xBTC gang on the Discord or Subreddit (/r/0xbitcoin).

Infernal Toast
Zegordo
Mikers
0x1d00ffff
Azlehria
LtTofu
... and more (LMK if I forgot someone!)


Please report your results with this build so that I can further refine it! I could particularly use feedback from users of
pre-Maxwell Gen2 GPUs! Thanks for trying out this build and for supporting 0xBitcoin! -- LtTofu / Mag517 (find me on Discord)