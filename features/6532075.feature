Feature: The user is able to view and filter recommendations

Scenario: User should be able to view recommendations
Given that I land on dell.com 
When I look at page
Then i should see recommendations

Scenario: User should be able to filter recommendations
Given that I land on dell.com 
When I enter mini and maxi
And I click ok
Then i should see recommendations