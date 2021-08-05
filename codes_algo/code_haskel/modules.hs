

-- MODULES P/87 --

import Data.List -- Imports all of Data.List's fxns.
--import Data.List (nub, sort) if you want to import particular fxns.

import Data.Char
--all imports should be above the rest of the code

--numUniques :: (Eq a) => [a] -> Int
--numUniques = length . nub

wordNums :: String -> [(String,Int)]
wordNums = map (\ph -> (head ph, length ph)). group . sort. words
--wordNums "wa wa wee wa"
--wordNums xs = map (\ph -> (head ph, length ph)) (group(sort (words (xs))))
--map (+3) [1..3] will give [4,5,6], i.e. map :: (a->b) -> [c] -> [c], as such map applies the fxn (\ph -> (head ph, length ph)) to every elmt. of (group(sort (words (xs)))).  

tmpMap :: (a->b) -> [a] -> [b]
--tmpMap takes a fxn., that takes value a and gives out value b, and gives out a fxn. that takes list of values a and gives out a list of values b. The fxn. (+3) is an example of a fxn. (a->b). Since the fxn (a->b) takes in values a, the list that is input is a list of values of a (i.e. [a]). 

tmpMap _ [] = []
tmpMap f (x:xs) = f x : tmpMap f xs -- there is pattern matching here.

-- Recall Pattern matching --
--tmpMap1 :: [a] -> a
--tmpMap1 (x:_) =  x

-- "hawaii" `isPrefixOf` "hawaii joe" 
encode :: Int -> String -> String
encode offset msg = map (\c -> chr $ ord c + offset) msg

digitSum :: Int -> Int
digitSum = sum . map digitToInt . show

firstTo40 :: Maybe Int
firstTo40 = find(\x -> digitSum x == 40) [1..40] 









