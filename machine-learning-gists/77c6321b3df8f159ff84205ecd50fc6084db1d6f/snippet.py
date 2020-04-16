#!/usr/bin/env python

# Source: http://www.codingninja.co.uk/best-programmers-quotes/

quotes = [ 
  {
    "text": "There are two ways of constructing a software design: One way is to make it so simple that there are obviously no deficiencies and the other way is to make it so complicated that there are no obvious deficiencies.",
    "author": "C.A.R. Hoare, The 1980 ACM Turing Award Lecture"
  }, {
    "text": "The computing scientist's main challenge is not to get confused by the complexities of his own making.",
    "author": "E. W. Dijkstra"
  }, {
    "text": "The cheapest, fastest, and most reliable components are those that aren't there.",
    "author": "Gordon Bell"
  }, {
    "text": "One of my most productive days was throwing away 1000 lines of code.",
    "author": "Ken Thompson"
  }, {
    "text": "When in doubt, use brute force.",
    "author": "Ken Thompson"
  }, {
    "text": "Deleted code is debugged code.",
    "author": "Jeff Sickel"
  }, {
    "text": "Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it.",
    "author": "Brian W. Kernighan and P. J. Plauger in The Elements of Programming Style."
  }, {
    "text": "The most effective debugging tool is still careful thought, coupled with judiciously placed print statements.",
    "author": "Brian W. Kernighan, in the paper Unix for Beginners (1979)"
  }, {
    "text": "Controlling complexity is the essence of computer programming.",
    "author": "Brian Kernighan"
  }, {
    "text": "Beauty is more important in computing than anywhere else in technology because software is so complicated. Beauty is the ultimate defence against complexity.",
    "author": "David Gelernter"
  }, {
    "text": "UNIX was not designed to stop its users from doing stupid things, as that would also stop them from doing clever things.",
    "author": "Doug Gwyn"
  }, {
    "text": "If you're willing to restrict the flexibility of your approach, you can almost always do something better.",
    "author": "John Carmack"
  }, {
    "text": "A data structure is just a stupid programming language.",
    "author": "R. Wm. Gosper"
  }, {
    "text": "The essence of XML is this: the problem it solves is not hard, and it does not solve the problem well.",
    "author": "Phil Wadler, POPL 2003"
  }, {
    "text": "A program that produces incorrect results twice as fast is infinitely slower.",
    "author": "John Osterhout"
  }, {
    "text": "Life is too short to run proprietary software.",
    "author": "Bdale Garbee"
  }, {
    "text": "Mathematicians stand on each others' shoulders and computer scientists stand on each others' toes.",
    "author": "Richard Hamming"
  }, {
    "text": "It's not that Perl programmers are idiots, it's that the language rewards idiotic behavior in a way that no other language or tool has ever done.",
    "author": "Erik Naggum, comp.lang.lisp"
  }, {
    "text": "It's a curious thing about our industry: not only do we not learn from our mistakes, we also don't learn from our successes.",
    "author": "Keith Braithwaite"
  }, {
    "text": "Ethernet always wins.",
    "author": "Andy Bechtolsheim"
  }, {
    "text": "The central enemy of reliability is complexity.",
    "author": "Geer et al."
  }, {
    "text": "Simplicity is prerequisite for reliability.",
    "author": "Edsger W. Dijkstra"
  }, {
    "text": "Beware of 'the real world'. A speaker's apeal to it is always an invitation not to challenge his tacit assumptions.",
    "author": "Edsger W. Dijkstra"
  }, {
    "text": "Unix is a junk OS designed by a committee of PhDs.",
    "author": "Dave Cutler"
  }, {
    "text": "i've wondered whether Linux sysfs should be called syphilis",
    "author": "forsyth"
  }, {
    "text": "Programming graphics in X is like finding the square root of PI using Roman numerals.",
    "author": "Henry Spencer"
  }, {
    "text": "Forward thinking was just the thing that made Multics what it is today.",
    "author": "Erik Quanstrom"
  }, {
    "text": "You want to make your way in the CS field? Simple. Calculate rough time of amnesia (hell, 10 years is plenty, probably 10 months is plenty), go to the dusty archives, dig out something fun, and go for it. It's worked for many people, and it can work for you.",
    "author": "Ron Minnich"
  }, {
    "text": "At first I hoped that such a technically unsound project would collapse but I soon realized it was doomed to success. Almost anything in software can be implemented, sold, and even used given enough determination. There is nothing a mere scientist can say that will stand against the flood of a hundred million dollars. But there is one quality that cannot be purchased in this way -and that is reliability. The price of reliability is the pursuit of the utmost simplicity. It is a price which the very rich find most hard to pay.",
    "author": "C.A.R. Hoare"
  }, {
    "text": "I remarked to Dennis [Ritchie] that easily half the code I was writing in Multics was error recovery code. He said, 'We left all that stuff out [of Unix]. If there's an error, we have this routine called panic, and when it is called, the machine crashes, and you holler down the hall, 'Hey, reboot it.'",
    "author": "Tom Van Vleck"
  }, {
    "text": "RMS is to Unix, like Hitler [was] to Nietzsche.",
    "author": "Federico Benavento"
  }, {
    "text": "Unix is simple. It just takes a genius to understand its simplicity.",
    "author": "Dennis Ritchie"
  }, {
    "text": "Most xml i've seen makes me think i'm dyslexic. it also looks constipated, and two health problems in one standard is just too much.",
    "author": "Charles Forsyth"
  }, {
    "text": "PHP is a minor evil perpetrated and created by incompetent amateurs, whereas Perl is a great and insidious evil perpetrated by skilled but perverted professionals.",
    "author": "Mike Stay"
  }, {
    "text": "This 'users are idiots, and are confused by functionality' mentality of Gnome is a disease. If you think your users are idiots, only idiots will use it.",
    "author": "Linus"
  }, {
    "text": "The key to performance is elegance, not battalions of special cases.",
    "author": "Jon Bentley and Doug McIlroy"
  }, {
    "text": "Just because the standard provides a cliff in front of you, you are not necessarily required to jump off it.",
    "author": "Norman Diamond"
  }, {
    "text": "Are you quite sure that all those bells and whistles, all those wonderful facilities of your so called powerful programming languages, belong to the solution set rather than the problem set?",
    "author": "Edsger W. Dijkstra"
  }, {
    "text": "Measuring programming progress by lines of code is like measuring aircraft building progress by weight.",
    "author": "Bill Gates"
  }, {
    "text": "The object-oriented model makes it easy to build up programs by accretion. What this often means, in practice, is that it provides a structured way to write spaghetti code.",
    "author": "Paul Graham"
  }, {
    "text": "First, solve the problem. Then, write the code.",
    "author": "John Johnson"
  }, {
    "text": "Most software today is very much like an Egyptian pyramid with millions of bricks piled on top of each other, with no structural integrity, but just done by brute force and thousands of slaves.",
    "author": "Alan Kay"
  }, {
    "text": "Correctness is clearly the prime quality. If a system does not do what it is supposed to do, then everything else about it matters little.",
    "author": "Bertrand Meyer"
  }, {
    "text": "Complexity kills. It sucks the life out of developers, it makes products difficult to plan, build and test, it introduces security challenges and it causes end-user and administrator frustration.",
    "author": "Ray Ozzie"
  }, {
    "text": "If the designers of X Windows built cars, there would be no fewer than five steering wheels hidden about the cockpit, none of which followed the same principles - but you'd be able to shift gears with your car stereo. Useful feature that.",
    "author": "Marcus J. Ranum, DEC"
  }, {
    "text": "A language that doesn't have everything is actually easier to program in than some that do.",
    "author": "Dennis M. Ritchie"
  }, {
    "text": "Mostly, when you see programmers, they aren't doing anything. One of the attractive things about programmers is that you cannot tell whether or not they are working simply by looking at them. Very often they're sitting there seemingly drinking coffee and gossiping, or just staring into space. What the programmer is trying to do is get a handle on all the individual and unrelated ideas that are scampering around in his head.",
    "author": "Charles M. Strauss"
  }, {
    "text": "Haskell is faster than C++, more concise than Perl, more regular than Python, more flexible than Ruby, more typeful than C#, more robust than Java, and has absolutely nothing in common with PHP.",
    "author": "Autrijus Tang"
  }, {
    "text": "You can't trust code that you did not totally create yourself.",
    "author": "Ken Thompson"
  }, {
    "text": "Object-oriented design is the roman numerals of computing.",
    "author": "Rob Pike"
  }, {
    "text": "Not only is UNIX dead, it's starting to smell really bad.",
    "author": "Rob Pike circa 1991"
  }, {
    "text": "We have persistant(sic) objects, they're called files.",
    "author": "Ken Thompson"
  }, {
    "text": "If you want to go somewhere, goto is the best way to get there.",
    "author": "ken"
  }, {
    "text": "The X server has to be the biggest program I've ever seen that doesn't do anything for you.",
    "author": "ken"
  }, {
    "text": "A smart terminal is not a smartass terminal, but rather a terminal you can educate.",
    "author": "Rob Pike"
  }, {
    "text": "Simplicity is the ultimate sophistication.",
    "author": "Leonardo da Vinci"
  }, {
    "text": "Increasingly, people seem to misinterpret complexity as sophistication, which is baffling-the incomprehensible should cause suspicion rather than admiration. Possibly this trend results from a mistaken belief that using a somewhat mysterious device confers an aura of power on the user.",
    "author": "Niklaus Wirth"
  }, {
    "text": "Compatibility means deliberately repeating other people's mistakes.",
    "author": "David Wheeler"
  }, {
    "text": "For the sinner deserves not life but death, according to the disk devices. For example, start with Plan 9, which is free of sin, the case is different from His perspective.",
    "author": "Mark V. Shaney"
  }, {
    "text": "Unix has retarded OS research by 10 years and linux has retarded it by 20.",
    "author": "Dennis Ritchie as quoted by by Boyd Roberts in 9fans."
  }, {
    "text": "Any program that tries to be so generalized and configurable that it could handle any kind of task will either fall short of this goal, or will be horribly broken.",
    "author": "Chris Wenham"
  }, {
    "text": "Nobody who uses XML knows what they are doing.",
    "author": "Chris Wenham"
  }, {
    "text": "Debugging time increases as a square of the program's size.",
    "author": "Chris Wenham"
  }, {
    "text": "in aeronautical circles, it's said that the f4 is proof that given enough thrust even a brick will fly.",
    "author": "erik quanstrom"
  }, {
    "text": "linux is the f4 of computing?",
    "author": "erik quanstrom"
  }, {
    "text": "Hofstadter's Law: It always takes longer than you expect, even when you take into account Hofstadter's Law.",
    "author": "P. J. Plauger, Computer Language, March 1983"
  }, {
    "text": "My definition of an expert in any field is a person who knows enough about what's really going on to be scared.",
    "author": "P. J. Plauger, Computer Language, March 1983"
  }, {
    "text": "Every language has an optimization operator. In C++ that operator is '//'",
    "author": "Linus Torvalds"
  }, {
    "text": "Nobody should start to undertake a large project. You start with a small trivial project, and you should never expect it to get large. If you do, you'll just overdesign and generally think it is more important than it likely is at that stage. Or worse, you might be scared away by the sheer size of the work you envision. So start small, and think about the details. Don't think about some big picture and fancy design. If it doesn't solve some fairly immediate need, it's almost certainly over-designed. And don't expect people to jump in and help you. That's not how these things work. You need to get something half-way useful first, and then others will say 'hey, that almost works for me', and they'll get involved in the project.",
    "author": "Linus Torvalds"
  }, {
    "text": "Theory is when you know something, but it doesn't work. Practice is when something works, but you don't know why. Programmers combine theory and practice: Nothing works and they don't know why.",
    "author": "David Parnas"
  }, {
    "text": "A computer is a stupid machine with the ability to do incredibly smart things, while computer programmers are smart people with the ability to do incredibly stupid things. They are, in short, a perfect match",
    "author": "David Parnas"
  }, {
    "text": "Q: What is the most often-overlooked risk in software engineering?\nA: Incompetent programmers. There are estimates that the number of programmers needed in the U.S. exceeds 200,000. This is entirely misleading. It is not a quantity problem; we have a quality problem. One bad programmer can easily create two new jobs a year. Hiring more bad programmers will just increase our perceived need for them. If we had more good programmers, and could easily identify them, we would need fewer, not more.",
    "author": "David Parnas"
  }, {
    "text": "Well over half of the time you spend working on a project (on the order of 70 percent) is spent thinking, and no tool, no matter how advanced, can think for you. Consequently, even if a tool did everything except the thinking for you - if it wrote 100 percent of the code, wrote 100 percent of the documentation, did 100 percent of the testing, burned the CD-ROMs, put them in boxes, and mailed them to your customers - the best you could hope for would be a 30 percent improvement in productivity. In order to do better than that, you have to change the way you think.",
    "author": "Dick Gabriel"
  }, {
    "text": "The best code is no code at all.",
    "author": "Dick Gabriel"
  }, {
    "text": "Before software can be reusable it first has to be usable.",
    "author": "Dick Gabriel"
  }, {
    "text": "Old programs read like quiet conversations between a well-spoken research worker and a well-studied mechanical colleague, not as a debate with a compiler. Who'd have guessed sophistication bought such noise?",
    "author": "Dick Gabriel"
  }, {
    "text": "More computing sins are committed in the name of efficiency (without necessarily achieving it) than for any other single reason - including blind stupidity.",
    "author": "William A. Wulf"
  }, {
    "text": "There is not now, nor has there ever been, nor will there ever be, any programming language in which it is the least bit difficult to write bad code.",
    "author": "Edsger W. Dijkstra"
  }, {
    "text": "Program testing can be a very effective way to show the presence of bugs, but is hopelessly inadequate for showing their absence.",
    "author": "Edsger W. Dijkstra"
  }, {
    "text": "The competent programmer is fully aware of the limited size of his own skull. He therefore approaches his task with full humility, and avoids clever tricks like the plague.",
    "author": "Edsger W. Dijkstra"
  }, {
    "text": "Parkinson's Law Otherwise known as the law of bureaucracy, this law states that... 'Work expands so as to fill the time available for its completion.'",
    "author": "Alan Cooper, About Face"
  }, {
    "text": "It has been said that the great scientific disciplines are examples of giants standing on the shoulders of other giants. It has also been said that the software industry is an example of midgets standing on the toes of other midgets.",
    "author": "Alan Cooper, About Face"
  }, {
    "text": "Code never lies, comments sometimes do.",
    "author": "Ron Jeffries"
  }, {
    "text": "What I cannot build, I do not understand.",
    "author": "Richard Feynman"
  }, {
    "text": "If we'd asked the customers what they wanted, they would have said 'faster horses'",
    "author": "Henry Ford"
  }, {
    "text": "I (...) am rarely happier than when spending an entire day programming my computer to perform automatically a task that would otherwise take me a good ten seconds to do by hand.",
    "author": "Douglas Adams, Last Chance to See"
  }, {
    "text": "Programming is not a zero-sum game. Teaching something to a fellow programmer doesn't take it away from you. I'm happy to share what I can, because I'm in it for the love of programming. The Ferraris are just gravy, honest!",
    "author": "John Carmack, from Michael Abrash' Graphics Programming Black Book."
  }, {
    "text": "I have found that the reason a lot of people are interested in artificial intelligence is the same reason a lot of people are interested in artificial limbs: they are missing one.",
    "author": "David Parnas"
  }, {
    "text": "Once you've dressed and before you leave the house, look in the mirror and take at least one thing off.",
    "author": "Coco Chanel"
  }, {
    "text": "When I am working on a problem I never think about beauty. I think only how to solve the problem. But when I have finished, if the solution is not beautiful, I know it is wrong.",
    "author": "R. Buckminster Fuller"
  }, {
    "text": "I have always found that plans are useless, but planning is indispensable.",
    "author": "Dwight D. Eisenhower"
  }, {
    "text": "I will, in fact, claim that the difference between a bad programmer and a good one is whether he considers his code or his data structures more important. Bad programmers worry about the code. Good programmers worry about data structures and their relationships.",
    "author": "Linus Torvalds"
  }, {
    "text": "Software is like entropy. It is difficult to grasp, weighs nothing, and obeys the second law of thermodynamics; i.e. it always increases.",
    "author": "u."
  }, {
    "text": "A fool with a tool is a more dangerous fool.",
    "author": "u."
  }, {
    "text": "Some problems are so complex that you have to be highly intelligent and well informed just to be undecided about them.",
    "author": "Laurence J. Peter"
  }, {
    "text": "The most amazing achievement of the computer software industry is its continuing cancellation of the steady and staggering gains made by the computer hardware industry.",
    "author": "Henry Petroski"
  }, {
    "text": "Theory is when you know something, but it doesn't work. Practice is when something works, but you don't know why. Programmers combine theory and practice: Nothing works and they don't know why.",
    "author": "Stewart Brand"
  }, {
    "text": "Once a new technology starts rolling, if you're not part of the steamroller, you're part of the road.",
    "author": "Stewart Brand"
  }, {
    "text": "Einstein argued that there must be simplified explanations of nature, because God is not capricious or arbitrary. No such faith comforts the software engineer.",
    "author": "Fred Brooks"
  }, {
    "text": "... the cost of adding a feature isn't just the time it takes to code it. The cost also includes the addition of an obstacle to future expansion. ... The trick is to pick the features that don't fight each other.",
    "author": "John Carmack"
  }, {
    "text": "With diligence it is possible to make anything run slowly.",
    "author": "Tom Duff"
  }, {
    "text": "Any intelligent fool can make things bigger, more complex, and more violent. It takes a touch of genius - and a lot of courage - to move in the opposite direction.",
    "author": "Albert Einstein"
  }, {
    "text": "A foolish consistency is the hobgoblin of little minds, adored by little statesmen and philosophers and divines.",
    "author": "Ralph Waldo Emerson"
  }, {
    "text": "For a sucessful technology, honesty must take precedence over public relations for nature cannot be fooled.",
    "author": "Richard Feynman"
  }, {
    "text": "Comparing to another activity is useful if it helps you formulate questions, it's dangerous when you use it to justify answers.",
    "author": "Martin Fowler"
  }, {
    "text": "Simplicity carried to the extreme becomes elegance.",
    "author": "Jon Franklin"
  }, {
    "text": "Software obeys the law of gaseous expansion - it continues to grow until memory is completely filled.",
    "author": "Larry Gleason"
  }, {
    "text": "The unavoidable price of reliability is simplicity.",
    "author": "C.A.R. Hoare"
  }, {
    "text": "The ability to simplify means to eliminate the unnecessary so that the necessary may speak.",
    "author": "Hans Hoffmann"
  }, {
    "text": "Trying to outsmart a compiler defeats much of the purpose of using one.",
    "author": "Kernighan and Plauger, The Elements of Programming Style."
  }, {
    "text": "You're bound to be unhappy if you optimize everything.",
    "author": "Donald Knuth"
  }, {
    "text": "A distributed system is one in which the failure of a computer you didn't even know existed can render your own computer unusable.",
    "author": "Leslie Lamport"
  }, {
    "text": "But in our enthusiasm, we could not resist a radical overhaul of the system, in which all of its major weaknesses have been exposed, analyzed, and replaced with new weaknesses.",
    "author": "Bruce Leverett, Register Allocation in Optimizing Compilers"
  }, {
    "text": "The proper use of comments is to compensate for our failure to express ourself in code.",
    "author": "Robert C. MartinClean Code"
  }, {
    "text": "If you want a product with certain characteristics, you must ensure that the team has those characteristics before the product's development.",
    "author": "Jim McCarthy and Michele McCarthy - Software for your Head"
  }, {
    "text": "You can't have great software without a great team, and most software teams behave like dysfunctional families.",
    "author": "Jim McCarthy"
  }, {
    "text": "Testing by itself does not improve software quality. Test results are an indicator of quality, but in and of themselves, they don't improve it. Trying to improve software quality by increasing the amount of testing is like trying to lose weight by weighing yourself more often. What you eat before you step onto the scale determines how much you will weigh, and the software development techniques you use determine how many errors testing will find. If you want to lose weight, don't buy a new scale; change your diet. If you want to improve your software, don't test more; develop better.",
    "author": "Steve McConnell Code Complete"
  }, {
    "text": "Correctness is clearly the prime quality. If a system does not do what it is supposed to do, then everything else about it matters little.",
    "author": "Bertrand Meyer"
  }, {
    "text": "Incorrect documentation is often worse than no documentation.",
    "author": "Bertrand Meyer"
  }, {
    "text": "Software sucks because users demand it to.",
    "author": "Nathan Myhrvold"
  }, {
    "text": "Unformed people delight in the gaudy and in novelty. Cooked people delight in the ordinary.",
    "author": "Erik Naggum"
  }, {
    "text": "There's no sense being exact about something if you don't even know what you're talking about.",
    "author": "John von Neumann"
  }, {
    "text": "That's the thing about people who think they hate computers. What they really hate is lousy programmers.",
    "author": "Larry Niven and Jerry Pournelle Oath of Fealty"
  }, {
    "text": "Search all the parks in all your cities; you'll find no statues of committees.",
    "author": "David Ogilvy"
  }, {
    "text": "Good code is short, simple, and symmetrical - the challenge is figuring out how to get there.",
    "author": "Sean Parent"
  }, {
    "text": "Fashion is something barbarous, for it produces innovation without reason and imitation without benefit.",
    "author": "George Santayana"
  }, {
    "text": "Forgive him, for he believes that the customs of his tribe are the laws of nature!",
    "author": "G.B. Shaw"
  }, {
    "text": "The only sin is to make a choice without knowing you are making one.",
    "author": "Jonathan Shewchuk"
  }, {
    "text": "It is a painful thing to look at your own trouble and know that you yourself and no one else has made it.",
    "author": "Sophocles, Ajax"
  }, {
    "text": "The primary duty of an exception handler is to get the error out of the lap of the programmer and into the surprised face of the user. Provided you keep this cardinal rule in mind, you can't go far wrong.",
    "author": "Verity Stob"
  }, {
    "text": "A notation is important for what it leaves out.",
    "author": "Joseph Stoy"
  }, {
    "text": "An organisation that treats its programmers as morons will soon have programmers that are willing and able to act like morons only.",
    "author": "Bjarne Stroustrup"
  }, {
    "text": "I have always wished that my computer would be as easy to use as my telephone. My wish has come true. I no longer know how to use my telephone.",
    "author": "Bjarne Stroustrup"
  }, {
    "text": "The most important single aspect of software development is to be clear about what you are trying to build.",
    "author": "Bjarne Stroustrup"
  }, {
    "text": "The best is the enemy of the good.",
    "author": "Voltaire"
  }, {
    "text": "As soon as we started programming, we found to our surprise that it wasn't as easy to get programs right as we had thought. Debugging had to be discovered. I can remember the exact instant when I realized that a large part of my life from then on was going to be spent in finding mistakes in my own programs.",
    "author": "Maurice Wilkes discovers debugging, 1949"
  }, {
    "text": "Software gets slower faster than hardware gets faster.",
    "author": "Wirth's law"
  }, {
    "text": "The purpose of software engineering is to control complexity, not to create it.",
    "author": "Dr. Pamela Zave"
  }, {
    "text": "I object to doing things that computers can do.",
    "author": "Olin Shivers"
  }, {
    "text": "Simplicity - the art of maximizing the amount of work not done - is essential.",
    "author": "From the Agile Manifesto."
  }, {
    "text": "When you want to do something differently from the rest of the world, it's a good idea to look into whether the rest of the world knows something you don't.",
    "author": "J.R.R. Tolkien"
  }, {
    "text": "Perilous to us all are the devices of an art deeper than that which we possess ourselves.",
    "author": "J.R.R. Tolkien"
  }, {
    "text": "Complexity has nothing to do with intelligence, simplicity does.",
    "author": "Larry Bossidy"
  }, {
    "text": "If it doesn't work, it doesn't matter how fast it doesn't work.",
    "author": "Mich Ravera"
  }, {
    "text": "Simplicity is hard to build, easy to use, and hard to charge for. Complexity is easy to build, hard to use, and easy to charge for.",
    "author": "Chris Sacca"
  }, {
    "text": "They won't tell you that they don't understand it; they will happily invent their way through the gaps and obscurities.",
    "author": "V.A. Vyssotsky on software programmers and their views on specifications"
  }, {
    "text": "In software, the most beautiful code, the most beautiful functions, and the most beautiful programs are sometimes not there at all.",
    "author": "Jon Bentley, Beautiful Code (O'Reilly), 'The Most Beautiful Code I Never Wrote'"
  }, {
    "text": "Computers make it easier to do a lot of things, but most of the things they make it easier to do don't need to be done.",
    "author": "Andy Rooney"
  }, {
    "text": "True glory consists in doing what deserves to be written; in writing what deserves to be read.",
    "author": "Pliny the Elder"
  }, {
    "text": "The whole point of getting things done is knowing what to leave undone.",
    "author": "Oswald Chambers"
  }, {
    "text": "The whole HTML validation exercise is questionable, but validating as XHTML is flat-out masochism. Only recommended for those that enjoy pain. Or programmers. I can't always tell the difference.",
    "author": "Jeff Atwood"
  }, {
    "text": "When in doubt, leave it out.",
    "author": "Joshua Bloch"
  }, {
    "text": "No code is faster than no code.",
    "author": "merb motto"
  }, {
    "text": "As a rule, software systems do not work well until they have been used, and have failed repeatedly, in real applications.",
    "author": "Dave Parnas"
  }, {
    "text": "OOP is to writing a program, what going through airport security is to flying.",
    "author": "Richard Mansfield"
  }, {
    "text": "The problem with object-oriented languages is they've got all this implicit environment that they carry around with them. You wanted a banana but what you got was a gorilla holding the banana and the entire jungle.",
    "author": "Joe Armstrong"
  }, {
    "text": "As a programmer, it is your job to put yourself out of business. What you do today can be automated tomorrow.",
    "author": "Doug McIlroy"
  }, {
    "text": "IDE features are language smells.",
    "author": "Reg Braithwaite"
  }, {
    "text": "A good way to have good ideas is by being unoriginal.",
    "author": "Bram Cohen"
  }, {
    "text": "a program is like a poem: you cannot write a poem without writing it. Yet people talk about programming as if it were a production process and measure 'programmer productivity'in terms of 'number of lines of code produced'.In so doing they book that number on the wrong side of the ledger: We should always refer to'the number of lines of code spent'.",
    "author": "E. W. Dijkstra"
  }, {
    "text": "'Layered approach' is not a magic incantation to excuse any bit of snake oil. Homeopathic remedies might not harm (pure water is pure water), but that's not an excuse for quackery. And frankly, most of the 'security improvement' crowd sound exactly like woo-peddlers.",
    "author": "Al Viro"
  }, {
    "text": "The trick is to fix the problem you have, rather than the problem you want.",
    "author": "Bram Cohen"
  }, {
    "text": "Security is a state of mind.",
    "author": "NSA Security Manual"
  }, {
    "text": "Never attribute to funny hardware that which can be adequately explained by broken locking.",
    "author": "Erik Quanstrom"
  }, {
    "text": "Things which any idiot could write usually have the quality of having been written by an idiot.",
    "author": "Bram Cohen"
  }, {
    "text": "In programming the hard part isn't solving problems, but deciding what problems to solve.",
    "author": "Paul Graham"
  }, {
    "text": "Do I really want to be using a language where memoize is a PhD-level topic?",
    "author": "Mark Engelberg about Haskell"
  }, {
    "text": "If you start programming by learning perl you will just become a menace to your self and others.",
    "author": "egoncasteel"
  }, {
    "text": "When there is no type hierarchy you don't have to manage the type hierarchy.",
    "author": "Rob Pike"
  }, {
    "text": "Programming languages should be designed not by piling feature on top of feature, but by removing the weaknesses and restrictions that make additional features appear necessary.",
    "author": "RnRS"
  }, {
    "text": "Software efficiency halves every 18 months, compensating Moore's Law.",
    "author": "May's Law"
  }, {
    "text": "So-called 'smart' software usually is the worst you can imagine.",
    "author": "Christian Neukirchen"
  }, {
    "text": "Such is modern computing: everything simple is made too complicated because it's easy to fiddle with; everything complicated stays complicated because it's hard to fix.",
    "author": "Rob Pike"
  }, {
    "text": "It is not that uncommon for the cost of an abstraction to outweigh the benefit it delivers. Kill one today!",
    "author": "John Carmack"
  }, {
    "text": "So much complexity in software comes from trying to make one thing do two things.",
    "author": "Ryan Singer"
  }, {
    "text": "The standard rule is, when you're in a hole, stop digging; that seems not to apply [to] software nowadays.",
    "author": "Ron Minnich"
  }, {
    "text": "Languages that try to disallow idiocy become themselves idiotic.",
    "author": "Rob Pike"
  }, {
    "text": "A complex system that works is invariably found to have evolved from a simple system that worked. The inverse proposition also appears to be true: A complex system designed from scratch never works and cannot be made to work.",
    "author": "John Gall"
  }, {
    "text": "'design patterns' are concepts used by people who can't learn by any method except memorization, so in place of actual programming ability, they memorize 'patterns' and throw each one in sequence at a problem until it works",
    "author": "Dark_Shikari"
  }, {
    "text": "One of the big lessons of a big project is you don't want people that aren't really programmers programming, you'll suffer for it!",
    "author": "John Carmack"
  }, {
    "text": "Premature optimization, that's like a sneeze. Premature abstraction is like ebola; it makes my eyes bleed.",
    "author": "Christer Ericson"
  }, {
    "text": "Premature optimizations can be troublesome to revert, but premature generalizations are often near impossible.",
    "author": "Emil Persson"
  }, {
    "text": "Premature optimization, that's like a fart. Premature abstraction is like taking a dump on another developer's desk.",
    "author": "Chris Eric"
  }, {
    "text": "Normal people believe that if it ain't broke, don't fix it. Engineers believe that if it ain't broke, it doesn't have enough features yet.",
    "author": "Scott Adams"
  }, {
    "text": "If you give someone a program, you will frustrate them for a day; if you teach them how to program, you will frustrate them for a lifetime.",
    "author": "David Leinweber (NOWS)"
  } 
]

import random
import subprocess

def random_quote():
    index = random.randrange(0, len(quotes) - 1)
    return quotes[index]

def format_quote(quote):
    text, author = quote["text"], quote["author"]
    return "{}\n - {}".format(text, author)

def main():
    quote = format_quote(random_quote())
    subprocess.call(["cowsay", "-f", "tux", quote])

if __name__ == "__main__":
    main()
